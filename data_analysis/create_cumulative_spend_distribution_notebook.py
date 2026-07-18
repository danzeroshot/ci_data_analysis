#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import html
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

INPUT_OVERRIDE = os.environ.get('CLUSTERED_CURVE_INPUT')
OUTPUT_SUFFIX = os.environ.get('CLUSTERED_CURVE_OUTPUT_SUFFIX', '').strip()


def scoped_path(path):
    path = Path(path)
    if not OUTPUT_SUFFIX:
        return path
    return path.with_name(f'{path.stem}_{OUTPUT_SUFFIX}{path.suffix}')


PREFERRED_INPUTS = ([Path(INPUT_OVERRIDE)] if INPUT_OVERRIDE else []) + [
    Path('clustered_data_input_allcustomers.csv'),
    Path('custpaydetails_clustered_cumulative_curves_allcustomers_2026-06-05-0923.csv'),
    Path('clustered_data_input.csv'),
    Path('custpaydetails_clustered_cumulative_curves.csv'),
    Path('ci_item_clustered_cumulative_curves.csv'),
]
OUT_PATH = scoped_path(Path('cumulative_spend_distribution_analysis.ipynb'))
SUMMARY_OUT = scoped_path(Path('cumulative_spend_distribution_summary.csv'))
BUCKET_OUT = scoped_path(Path('cumulative_spend_distribution_by_elapsed_bucket.csv'))


def pick_input():
    for p in PREFERRED_INPUTS:
        if p.exists():
            return p
    raise FileNotFoundError('Expected clustered_data_input.csv or clustered cumulative curve CSV')


def norm_key(row, *names):
    upper = {k.upper(): v for k, v in row.items()}
    compact = {k.upper().replace('_', ''): v for k, v in row.items()}
    for name in names:
        if name in row:
            return row[name]
        if name.upper() in upper:
            return upper[name.upper()]
        key = name.upper().replace('_', '')
        if key in compact:
            return compact[key]
    return ''


def load_points(path):
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            customer = norm_key(r, 'CustomerName', 'CUSTOMERNAME') or 'Unknown'
            item = norm_key(r, 'ITEMID')
            split = norm_key(r, 'TrainSplit', 'TRAIN_SPLIT') or ('test' if hash((customer, item)) % 5 == 0 else 'train')
            rows.append({
                'customer': customer,
                'item': item,
                'split': split.lower(),
                'x': float(norm_key(r, 'ElapsedPct', 'ELAPSED_PCT')),
                'y': float(norm_key(r, 'CumulativeBurnPct', 'CUMULATIVE_BURN_PCT')),
                'total': float(norm_key(r, 'ItemTotalBurn', 'ITEM_TOTAL_BURN') or 0),
                'clusters': int(float(norm_key(r, 'ItemClusterCount', 'ITEM_CLUSTER_COUNT') or 0)),
                'days': float(norm_key(r, 'ItemModeledDays', 'ITEM_MODELED_DAYS') or 0),
                'seq': int(float(norm_key(r, 'ClusterSequence', 'CLUSTER_SEQUENCE') or 0)),
                'cluster_burn': float(norm_key(r, 'ClusterBurn', 'CLUSTER_BURN') or 0),
            })
    return rows


def beta_cdf(x, a, b):
    x = np.clip(np.asarray(x, dtype=float), 1e-9, 1 - 1e-9)
    return stats.beta.cdf(x, a, b)


def fit_beta_cdf(x, y):
    x = np.asarray(x, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 0, 1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    def loss(theta):
        a, b = np.exp(theta)
        pred = beta_cdf(x, a, b)
        return float(np.mean((pred - y) ** 2))

    best = None
    for start in [(1, 1), (.5, .8), (.7, 1.4), (1.4, .7), (2, 2), (.35, 1.4), (1.4, .35), (3, 1), (1, 3)]:
        res = optimize.minimize(loss, np.log(np.array(start)), method='Nelder-Mead', options={'maxiter': 1400})
        if best is None or res.fun < best.fun:
            best = res
    a, b = np.exp(best.x)
    pred = beta_cdf(x, a, b)
    err = pred - y
    return {
        'alpha': float(a),
        'beta': float(b),
        'mse': float(best.fun),
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'bias': float(np.mean(err)),
    }


def anchored_design(x, degree):
    x = np.asarray(x, dtype=float)
    cols = [x + 0.0]
    for power in range(degree - 1):
        cols.append(x * (1 - x) * (x ** power))
    return np.vstack(cols).T


def fit_poly_curve(x, y, degree=4):
    x = np.asarray(x, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 0, 1)
    coef, *_ = np.linalg.lstsq(anchored_design(x, degree), y, rcond=None)
    pred = np.clip(anchored_design(x, degree).dot(coef), 0, 1)
    err = pred - y
    grid = np.linspace(0, 1, 501)
    raw_grid = anchored_design(grid, degree).dot(coef)
    return {
        'degree': degree,
        'coef': coef,
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'bias': float(np.mean(err)),
        'clip_share_grid': float(np.mean((raw_grid < 0) | (raw_grid > 1))),
        'monotonic_violations': int(np.sum(np.diff(np.clip(raw_grid, 0, 1)) < -1e-6)),
    }


def pred_poly(fit, x):
    return np.clip(anchored_design(np.asarray(x, dtype=float), fit['degree']).dot(fit['coef']), 0, 1)


def bucket_duration(days):
    if days <= 180:
        return '<=180d'
    if days <= 365:
        return '181-365d'
    if days <= 730:
        return '366-730d'
    return '>730d'


def bucket_clusters(n):
    if n <= 3:
        return '<=3 clusters'
    if n <= 6:
        return '4-6 clusters'
    if n <= 12:
        return '7-12 clusters'
    return '13+ clusters'


def elapsed_bucket(x):
    idx = min(9, max(0, int(float(x) * 10)))
    return idx, idx / 10, (idx + 1) / 10


def fmt(x, p=4):
    return f'{float(x):,.{p}f}'


def fmt_int(x):
    return f'{int(x):,}'


def pct(x):
    return f'{100 * float(x):.2f}%'


def esc(x):
    return str(x).replace('|', '\\|').replace('\n', '<br>')


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(esc(v) for v in row) + ' |')
    return '\n'.join(out)


def md_cell(s):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': s}


def code_cell(s):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s}


def svg_uri(svg):
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode('ascii')


def xml_escape(value):
    return html.escape(str(value), quote=False)


def plot_overall_hist(y):
    width, height = 920, 500
    ml, mr, mt, mb = 70, 25, 34, 58
    bins = np.linspace(0, 1, 61)
    hist, edges = np.histogram(np.clip(y, 0, 1), bins=bins)
    ymax = hist.max() * 1.16
    def sx(x): return ml + x * (width - ml - mr)
    def sy(v): return height - mb - v / ymax * (height - mt - mb)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
    for frac in [0, .25, .5, .75, 1]:
        yy = sy(ymax * frac)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{width-mr}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{int(ymax*frac)}</text>')
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        x = sx(lo); bw = max(1, sx(hi)-sx(lo)-1); yy = sy(h)
        parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{height-mb-yy:.1f}" fill="#60a5fa" opacity="0.62"/>')
    for q, color in [(np.quantile(y, .25), '#059669'), (np.quantile(y, .5), '#111827'), (np.quantile(y, .75), '#059669')]:
        parts.append(f'<line x1="{sx(q):.1f}" y1="{mt}" x2="{sx(q):.1f}" y2="{height-mb}" stroke="{color}" stroke-width="2" stroke-dasharray="5 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Overall cumulative spend percent distribution</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">Cumulative spend percent</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_heatmap(x, y):
    width, height = 880, 570
    ml, mr, mt, mb = 70, 95, 34, 62
    h, xedges, yedges = np.histogram2d(np.clip(x, 0, 1), np.clip(y, 0, 1), bins=[30, 30])
    vmax = np.quantile(h[h > 0], .95) if np.any(h > 0) else 1
    def sx(v): return ml + v * (width - ml - mr)
    def sy(v): return height - mb - v * (height - mt - mb)
    def color(c):
        z = min(1, c / vmax) if vmax else 0
        r = int(245 - 185 * z); g = int(247 - 115 * z); b = int(250 - 5 * z)
        return f'rgb({r},{g},{b})'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            c = h[i, j]
            if c <= 0:
                continue
            x0, x1 = xedges[i], xedges[i+1]
            y0, y1 = yedges[j], yedges[j+1]
            parts.append(f'<rect x="{sx(x0):.1f}" y="{sy(y1):.1f}" width="{sx(x1)-sx(x0)+.2:.1f}" height="{sy(y0)-sy(y1)+.2:.1f}" fill="{color(c)}"/>')
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#d1d5db" opacity="0.7"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{width-mr}" y2="{sy(t):.1f}" stroke="#d1d5db" opacity="0.7"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    # legend
    lx, ly = width - 68, mt + 20
    for k in range(80):
        z = k / 79
        c = color(z * vmax)
        parts.append(f'<rect x="{lx}" y="{ly + 180 - k*2.2:.1f}" width="16" height="2.5" fill="{c}"/>')
    parts.append(f'<text x="{lx+24}" y="{ly+5}" font-size="11">dense</text><text x="{lx+24}" y="{ly+184}" font-size="11">sparse</text>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Joint distribution: elapsed percent vs cumulative spend percent</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">Elapsed percent</text>')
    parts.append(f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13">Cumulative spend percent</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def bucket_stats(rows):
    groups = defaultdict(list)
    for r in rows:
        idx, lo, hi = elapsed_bucket(r['x'])
        groups[idx].append(r)
    out = []
    for idx in range(10):
        pts = groups[idx]
        if not pts:
            continue
        vals = np.clip(np.array([p['y'] for p in pts]), 0, 1)
        xs = np.array([p['x'] for p in pts])
        lo, hi = idx / 10, (idx + 1) / 10
        out.append({
            'bucket': f'{lo:.1f}-{hi:.1f}',
            'count': len(vals),
            'x_mid': float(np.mean(xs)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'p05': float(np.quantile(vals, .05)),
            'p10': float(np.quantile(vals, .10)),
            'p25': float(np.quantile(vals, .25)),
            'median': float(np.quantile(vals, .50)),
            'p75': float(np.quantile(vals, .75)),
            'p90': float(np.quantile(vals, .90)),
            'p95': float(np.quantile(vals, .95)),
            'iqr': float(np.quantile(vals, .75) - np.quantile(vals, .25)),
            'skew': float(stats.skew(vals, bias=False)) if len(vals) > 2 else 0,
        })
    return out


def plot_percentile_bands(stats_rows, beta_fit, poly_fit):
    width, height = 920, 560
    ml, mr, mt, mb = 70, 26, 34, 62
    def sx(x): return ml + x * (width - ml - mr)
    def sy(y): return height - mb - y * (height - mt - mb)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{width-mr}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    xs = np.array([r['x_mid'] for r in stats_rows])
    for lo_key, hi_key, color, op in [('p05', 'p95', '#bfdbfe', .65), ('p25', 'p75', '#60a5fa', .50)]:
        upper = [(x, r[hi_key]) for x, r in zip(xs, stats_rows)]
        lower = [(x, r[lo_key]) for x, r in zip(xs[::-1], stats_rows[::-1])]
        pts = ' '.join([f'{sx(x):.1f},{sy(y):.1f}' for x, y in upper + lower])
        parts.append(f'<polygon points="{pts}" fill="{color}" opacity="{op}"/>')
    for key, color, width_line in [('median', '#111827', 3), ('mean', '#f97316', 2)]:
        parts.append('<polyline points="' + ' '.join(f'{sx(r["x_mid"]):.1f},{sy(r[key]):.1f}' for r in stats_rows) + f'" fill="none" stroke="{color}" stroke-width="{width_line}"/>')
    grid = np.linspace(0, 1, 240)
    beta_y = beta_cdf(grid, beta_fit['alpha'], beta_fit['beta'])
    poly_y = pred_poly(poly_fit, grid)
    parts.append('<polyline points="' + ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(grid, beta_y)) + '" fill="none" stroke="#dc2626" stroke-width="3"/>')
    parts.append('<polyline points="' + ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(grid, poly_y)) + '" fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#6b7280" stroke-width="2" stroke-dasharray="4 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Conditional cumulative spend distribution by elapsed percent</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">Elapsed percent</text>')
    legend = [('5-95% band', '#bfdbfe'), ('25-75% band', '#60a5fa'), ('Median', '#111827'), ('Mean', '#f97316'), ('Beta CDF', '#dc2626'), ('Anchored polynomial', '#059669'), ('Linear', '#6b7280')]
    for i, (label, color) in enumerate(legend):
        yy = 66 + i * 22
        if 'band' in label:
            parts.append(f'<rect x="650" y="{yy-10}" width="24" height="10" fill="{color}" opacity="0.65"/>')
        else:
            parts.append(f'<line x1="648" y1="{yy-5}" x2="676" y2="{yy-5}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="684" y="{yy}" font-size="12">{label}</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_bucket_distributions(rows):
    width, height = 980, 620
    ml, mr, mt, mb = 88, 25, 34, 74
    bins = np.linspace(0, 1, 26)
    hists = []
    maxh = 1
    for idx in range(10):
        vals = [r['y'] for r in rows if elapsed_bucket(r['x'])[0] == idx]
        hist, edges = np.histogram(np.clip(vals, 0, 1), bins=bins)
        hists.append(hist)
        maxh = max(maxh, hist.max())
    row_h = (height - mt - mb) / 10
    def sx(v): return ml + v * (width - ml - mr)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for idx, hist in enumerate(hists):
        y0 = mt + idx * row_h
        center = y0 + row_h * .55
        parts.append(f'<text x="{ml-12}" y="{center+4:.1f}" text-anchor="end" font-size="12">{idx/10:.1f}-{(idx+1)/10:.1f}</text>')
        parts.append(f'<line x1="{ml}" y1="{y0+row_h:.1f}" x2="{width-mr}" y2="{y0+row_h:.1f}" stroke="#f1f5f9"/>')
        for c, lo, hi in zip(hist, bins[:-1], bins[1:]):
            if c == 0:
                continue
            bar_h = (c / maxh) * row_h * .78
            parts.append(f'<rect x="{sx(lo):.1f}" y="{center-bar_h/2:.1f}" width="{sx(hi)-sx(lo)-1:.1f}" height="{bar_h:.1f}" fill="#2563eb" opacity="0.52"/>')
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#d1d5db"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-38}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Distribution of cumulative spend percent within elapsed-percent buckets</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13">Cumulative spend percent within each elapsed bucket</text>')
    parts.append(f'<text x="22" y="{height/2}" transform="rotate(-90 22 {height/2})" text-anchor="middle" font-size="13">Elapsed percent bucket</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_residual_hist(y, x, beta_fit, poly_fit):
    residuals = {
        'Linear': y - x,
        'Beta CDF': y - beta_cdf(x, beta_fit['alpha'], beta_fit['beta']),
        'Polynomial': y - pred_poly(poly_fit, x),
    }
    width, height = 920, 500
    ml, mr, mt, mb = 70, 25, 34, 58
    bins = np.linspace(-0.8, 0.8, 65)
    hists = {k: np.histogram(v, bins=bins)[0] for k, v in residuals.items()}
    ymax = max(h.max() for h in hists.values()) * 1.15
    def sx(v): return ml + (v + .8) / 1.6 * (width - ml - mr)
    def sy(v): return height - mb - v / ymax * (height - mt - mb)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [-.5, -.25, 0, .25, .5]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-35}" text-anchor="middle" font-size="12">{t:+.2f}</text>')
    colors = {'Linear': '#111827', 'Beta CDF': '#dc2626', 'Polynomial': '#059669'}
    for label, hist in hists.items():
        pts = []
        for c, lo, hi in zip(hist, bins[:-1], bins[1:]):
            pts.append(f'{sx((lo+hi)/2):.1f},{sy(c):.1f}')
        parts.append('<polyline points="' + ' '.join(pts) + f'" fill="none" stroke="{colors[label]}" stroke-width="3" opacity="0.75"/>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{mt}" x2="{sx(0):.1f}" y2="{height-mb}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Position residual distributions under alternative reference curves</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">Actual cumulative pct - expected cumulative pct</text>')
    for i, (label, color) in enumerate(colors.items()):
        yy = 70 + i * 24
        parts.append(f'<line x1="660" y1="{yy}" x2="690" y2="{yy}" stroke="{color}" stroke-width="3"/><text x="700" y="{yy+4}" font-size="12">{label}</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_stratified_curves(rows, stratifier, title):
    width, height = 900, 540
    ml, mr, mt, mb = 70, 26, 34, 62
    groups = defaultdict(list)
    for r in rows:
        groups[stratifier(r)].append(r)
    colors = ['#dc2626', '#2563eb', '#059669', '#f97316', '#7c3aed', '#111827']
    def sx(x): return ml + x * (width - ml - mr)
    def sy(y): return height - mb - y * (height - mt - mb)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{width-mr}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    for idx, key in enumerate(sorted(groups.keys())):
        pts = groups[key]
        if len(pts) < 20:
            continue
        stat_rows = bucket_stats(pts)
        color = colors[idx % len(colors)]
        parts.append('<polyline points="' + ' '.join(f'{sx(r["x_mid"]):.1f},{sy(r["median"]):.1f}' for r in stat_rows) + f'" fill="none" stroke="{color}" stroke-width="3"/>')
        yy = 68 + idx * 24
        parts.append(f'<line x1="642" y1="{yy}" x2="672" y2="{yy}" stroke="{color}" stroke-width="3"/><text x="682" y="{yy+4}" font-size="12">{xml_escape(key)} ({len(pts):,})</text>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#6b7280" stroke-width="2" stroke-dasharray="4 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">{xml_escape(title)}</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">Elapsed percent</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def analyze(rows):
    train = [r for r in rows if r['split'] != 'test']
    if not train:
        train = rows
    x = np.array([r['x'] for r in train], dtype=float)
    y = np.clip(np.array([r['y'] for r in train], dtype=float), 0, 1)
    all_x = np.array([r['x'] for r in rows], dtype=float)
    all_y = np.clip(np.array([r['y'] for r in rows], dtype=float), 0, 1)
    beta_fit = fit_beta_cdf(x, y)
    poly3 = fit_poly_curve(x, y, 3)
    poly4 = fit_poly_curve(x, y, 4)
    poly_fit = min([poly3, poly4], key=lambda f: (f['monotonic_violations'], f['clip_share_grid'], f['rmse']))
    stats_rows = bucket_stats(train)
    return train, x, y, all_x, all_y, beta_fit, poly3, poly4, poly_fit, stats_rows


def build_notebook(path, rows):
    train, x, y, all_x, all_y, beta_fit, poly3, poly4, poly_fit, stats_rows = analyze(rows)
    item_count = len({(r['customer'], r['item']) for r in rows})
    train_items = len({(r['customer'], r['item']) for r in train})
    complete_edge_share = float(np.mean(all_y >= .999999))
    zero_edge_share = float(np.mean(all_y <= .000001))
    corr = float(np.corrcoef(all_x, all_y)[0, 1])
    spearman = float(stats.spearmanr(all_x, all_y).correlation)

    overall = [
        ['Input CSV', path.name],
        ['Cluster curve rows', fmt_int(len(rows))],
        ['Items', fmt_int(item_count)],
        ['Train rows used for distribution fitting', fmt_int(len(train))],
        ['Train items', fmt_int(train_items)],
        ['Mean cumulative spend pct', pct(np.mean(all_y))],
        ['Median cumulative spend pct', pct(np.median(all_y))],
        ['Std dev cumulative spend pct', pct(np.std(all_y))],
        ['Share at completed edge near 100%', pct(complete_edge_share)],
        ['Share at zero edge near 0%', pct(zero_edge_share)],
        ['Pearson corr elapsed vs cumulative spend', fmt(corr, 4)],
        ['Spearman corr elapsed vs cumulative spend', fmt(spearman, 4)],
    ]
    quantile_rows = [[f'p{int(q*100):02d}', pct(np.quantile(all_y, q))] for q in [.01, .05, .10, .25, .50, .75, .90, .95, .99]]
    fit_rows = [
        ['Beta CDF', fmt(beta_fit['alpha'], 6), fmt(beta_fit['beta'], 6), fmt(beta_fit['mae'], 4), fmt(beta_fit['rmse'], 4), fmt(beta_fit['bias'], 4), '', ''],
        ['Anchored polynomial degree 3', ', '.join(fmt(c, 6) for c in poly3['coef']), '', fmt(poly3['mae'], 4), fmt(poly3['rmse'], 4), fmt(poly3['bias'], 4), fmt(poly3['clip_share_grid'], 4), fmt_int(poly3['monotonic_violations'])],
        ['Anchored polynomial degree 4', ', '.join(fmt(c, 6) for c in poly4['coef']), '', fmt(poly4['mae'], 4), fmt(poly4['rmse'], 4), fmt(poly4['bias'], 4), fmt(poly4['clip_share_grid'], 4), fmt_int(poly4['monotonic_violations'])],
    ]
    bucket_rows = [[r['bucket'], fmt_int(r['count']), pct(r['mean']), pct(r['median']), pct(r['p10']), pct(r['p25']), pct(r['p75']), pct(r['p90']), pct(r['iqr']), fmt(r['skew'], 3)] for r in stats_rows]
    duration_plot = plot_stratified_curves(train, lambda r: bucket_duration(r['days']), 'Median cumulative spend curves by item duration bucket')
    cluster_plot = plot_stratified_curves(train, lambda r: bucket_clusters(r['clusters']), 'Median cumulative spend curves by item cluster-count bucket')

    with SUMMARY_OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for row in overall:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(['model', 'alpha_or_coefficients', 'beta', 'mae', 'rmse', 'bias', 'clip_share_grid', 'monotonic_violations'])
        writer.writerows(fit_rows)
    with BUCKET_OUT.open('w', newline='', encoding='utf-8') as f:
        fieldnames = list(stats_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    cells = [
        md_cell('# Cumulative Spend Distribution Analysis\n\nThis notebook characterizes the underlying cumulative spend-position distribution that feeds the later Beta CDF modeling work. It should be read before `clustered_curve_beta_cdf_model.ipynb`: this page explains the empirical shape of `CUMULATIVEBURNPCT` as a function of `ELAPSEDPCT`, while the later notebook evaluates predictive reference curves against held-out items.'),
        md_cell('## Dataset and Overall Character\n\n' + table(['Metric', 'Value'], overall)),
        code_cell("import csv\nfrom pathlib import Path\n\npath = Path('clustered_data_input.csv')\nwith path.open(newline='', encoding='utf-8-sig') as handle:\n    reader = csv.DictReader(handle)\n    rows = list(reader)\nprint(path)\nprint(len(rows))\nprint(reader.fieldnames)"),
        md_cell('## Marginal Distribution of Cumulative Spend\n\nThe marginal distribution is bounded between 0 and 1 and is strongly affected by repeated observations from the same item curve. It is not a standalone time-free distribution: a point at 90% elapsed and a point at 20% elapsed should not be expected to have the same cumulative spend behavior. The visible pile-up near 100% is expected because every completed item contributes a final cluster at full cumulative spend.\n\n' + table(['Quantile', 'Cumulative spend pct'], quantile_rows) + '\n\n<img src="' + plot_overall_hist(all_y) + '" />'),
        md_cell('## Joint Distribution With Elapsed Percent\n\nThe joint view is the core characterization. The distribution is bounded, monotonic in expectation, heteroscedastic, and edge-inflated near completion. The diagonal line is the pure linear spend reference; density above the line is front-loaded spend, while density below it is back-loaded or delayed spend.\n\n<img src="' + plot_heatmap(all_x, all_y) + '" />'),
        md_cell('## Conditional Distribution by Elapsed Bucket\n\nThe table and band plot summarize `CUMULATIVEBURNPCT | ELAPSEDPCT bucket`. The widening and narrowing of the percentile bands matters more than the overall histogram because the production question is where an item sits relative to expected cumulative spend at its current elapsed position.\n\n' + table(['Elapsed bucket', 'Rows', 'Mean', 'Median', 'P10', 'P25', 'P75', 'P90', 'IQR', 'Skew'], bucket_rows)),
        md_cell('## Percentile Bands and Reference Curves\n\nThe empirical median curve is generally above the linear reference for much of the item life, which means these clustered payment curves are often front-loaded relative to time. The Beta CDF and anchored polynomial are compact parameterizations of that empirical shape; the later model notebook evaluates the Beta CDF approach more formally.\n\n<img src="' + plot_percentile_bands(stats_rows, beta_fit, poly_fit) + '" />'),
        md_cell('## Bucketed Density View\n\nThis view shows the full shape within each elapsed bucket rather than only percentiles. It makes the conditional spread visible: some elapsed regions are broad and multi-modal because items can receive lumpy clustered payments at different points in their lifecycle.\n\n<img src="' + plot_bucket_distributions(train) + '" />'),
        md_cell('## Curve Parameterizations\n\n' + table(['Model', 'Alpha / coefficients', 'Beta', 'MAE', 'RMSE', 'Bias', 'Clip share', 'Monotonic violations'], fit_rows) + '\n\nThe Beta CDF is a useful production candidate because it is naturally bounded and monotone. The anchored polynomial is useful as a flexible descriptive reference, but it is less constrained structurally and should be treated as diagnostic unless monotonicity and stability are explicitly checked.'),
        md_cell('## Residual Distribution Around Reference Curves\n\nResiduals are `actual cumulative pct - expected cumulative pct`. A good reference curve centers this distribution closer to zero and reduces asymmetric bias. The residual plot shows why the cumulative spend model should not be purely linear: the linear reference leaves a larger systematic position offset where the empirical spend curve is front-loaded.\n\n<img src="' + plot_residual_hist(y, x, beta_fit, poly_fit) + '" />'),
        md_cell('## Stratification by Duration\n\nDuration buckets have visibly different median cumulative spend curves. This supports the later duration-bucket Beta CDF model: the expected curve is not fully universal across short and long items.\n\n<img src="' + duration_plot + '" />'),
        md_cell('## Stratification by Cluster Count\n\nCluster count is another proxy for payment cadence and lumpiness. Items with fewer clusters tend to show coarser jumps, while higher-cluster items provide smoother cumulative curves.\n\n<img src="' + cluster_plot + '" />'),
        md_cell('## Interpretation\n\nThe cumulative spend distribution is best understood as a conditional bounded distribution, not as a single ordinary marginal distribution. Its important properties are:\n\n- Bounded support at `[0, 1]`, with a structural edge at 100% from final clusters.\n- Strong positive relationship between elapsed percent and cumulative spend percent.\n- A median curve that is not purely linear, indicating systematic front-loading in the clustered payment data.\n- Wide conditional dispersion, especially in the middle of the lifecycle, caused by lumpy payment postings and differing payment cadence.\n- Meaningful stratification by item duration and cluster count.\n\nThis motivates the next-stage notebook: fit expected-position curves and evaluate whether Beta CDF parameterizations outperform a transparent linear reference.'),
    ]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    OUT_PATH.write_text(json.dumps(nb, indent=2), encoding='utf-8')


def main():
    path = pick_input()
    rows = load_points(path)
    build_notebook(path, rows)
    print(f'wrote {OUT_PATH} using {path}')
    print(f'wrote {SUMMARY_OUT}')
    print(f'wrote {BUCKET_OUT}')


if __name__ == '__main__':
    main()
