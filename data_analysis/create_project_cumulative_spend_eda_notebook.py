#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import html
import json
import math
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

CSV_PATH = Path('custpaydetails_project_clustered_cumulative_curves_allcustomers_2026-06-08-1000.csv')
OUT_PATH = Path('project_cumulative_spend_eda_allcustomers.ipynb')
PROFILE_OUT = Path('project_cumulative_spend_eda_profile_allcustomers.csv')
MODEL_OUT = Path('project_cumulative_spend_model_family_assessment_allcustomers.csv')
BUCKET_OUT = Path('project_cumulative_spend_elapsed_bucket_summary_allcustomers.csv')


def md_cell(source):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': source}


def code_cell(source):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}


def esc(value):
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def xml(value):
    return html.escape(str(value), quote=False)


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(esc(v) for v in row) + ' |')
    return '\n'.join(out)


def svg_uri(svg):
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode('utf-8')).decode('ascii')


def fmt(x, p=4):
    if x is None or not np.isfinite(float(x)):
        return ''
    return f'{float(x):,.{p}f}'


def fmt_int(x):
    return f'{int(x):,}'


def pct(x):
    return f'{100 * float(x):.2f}%'


def fnum(row, key):
    val = row.get(key, '')
    if val is None or val == '':
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan


def load_rows():
    rows = []
    with CSV_PATH.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for r in reader:
            rows.append({
                'customer': r.get('CUSTOMERNAME', ''),
                'project': r.get('PROJECTNAME', ''),
                'split': (r.get('TRAINSPLIT', '') or '').lower(),
                'seq': fnum(r, 'CLUSTERSEQUENCE'),
                'x': fnum(r, 'ELAPSEDPCT'),
                'y_raw': fnum(r, 'CUMULATIVEBURNPCT'),
                'y': np.clip(fnum(r, 'CUMULATIVEBURNPCT'), 0, 1) if np.isfinite(fnum(r, 'CUMULATIVEBURNPCT')) else np.nan,
                'cluster_burn': fnum(r, 'CLUSTERBURN'),
                'total': fnum(r, 'PROJECTTOTALBURN'),
                'clusters': fnum(r, 'PROJECTCLUSTERCOUNT'),
                'days': fnum(r, 'PROJECTMODELEDDAYS'),
                'span': fnum(r, 'PROJECTSPANDAYS'),
                'paygroups': fnum(r, 'NUMPAYGROUPS'),
                'rows_in_cluster': fnum(r, 'ROWSINCLUSTER'),
            })
    return rows, fields


def group_projects(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['customer'], r['project'])].append(r)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda r: (r['seq'] if np.isfinite(r['seq']) else 0))
    return grouped


def bucket_cluster_count(n):
    if n <= 1: return '1 cluster'
    if n <= 2: return '2 clusters'
    if n <= 3: return '3 clusters'
    if n <= 6: return '4-6 clusters'
    if n <= 12: return '7-12 clusters'
    if n <= 24: return '13-24 clusters'
    return '25+ clusters'


def bucket_duration(days):
    if days <= 60: return '<=60d'
    if days <= 180: return '61-180d'
    if days <= 365: return '181-365d'
    if days <= 730: return '366-730d'
    return '>730d'


def bucket_size(total):
    if total <= 100000: return '<=100k'
    if total <= 500000: return '100k-500k'
    if total <= 1000000: return '500k-1m'
    if total <= 5000000: return '1m-5m'
    if total <= 25000000: return '5m-25m'
    return '>25m'


def elapsed_bucket(x):
    idx = min(9, max(0, int(float(x) * 10)))
    return idx, idx / 10, (idx + 1) / 10


def pava(y, weights=None):
    y = np.asarray(y, dtype=float)
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    levels = []
    level_weights = []
    counts = []
    for yi, wi in zip(y, weights):
        levels.append(float(yi)); level_weights.append(float(wi)); counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            w = level_weights[-2] + level_weights[-1]
            level = (levels[-2] * level_weights[-2] + levels[-1] * level_weights[-1]) / w
            c = counts[-2] + counts[-1]
            levels[-2:] = [level]
            level_weights[-2:] = [w]
            counts[-2:] = [c]
    return np.repeat(levels, counts)


def empirical_bucket_stats(points, bins=20):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        vals = [p['y'] for p in points if np.isfinite(p['x']) and np.isfinite(p['y']) and p['x'] >= lo and (p['x'] <= hi if hi == 1 else p['x'] < hi)]
        if not vals:
            continue
        arr = np.asarray(vals)
        rows.append({
            'bucket': f'{lo:.2f}-{hi:.2f}',
            'x_mid': (lo + hi) / 2,
            'count': len(arr),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p05': float(np.quantile(arr, .05)),
            'p10': float(np.quantile(arr, .10)),
            'p25': float(np.quantile(arr, .25)),
            'p75': float(np.quantile(arr, .75)),
            'p90': float(np.quantile(arr, .90)),
            'p95': float(np.quantile(arr, .95)),
            'iqr': float(np.quantile(arr, .75) - np.quantile(arr, .25)),
        })
    return rows


def fit_family_models(train, test):
    tx = np.array([r['x'] for r in train if np.isfinite(r['x']) and np.isfinite(r['y'])])
    ty = np.array([r['y'] for r in train if np.isfinite(r['x']) and np.isfinite(r['y'])])
    vx = np.array([r['x'] for r in test if np.isfinite(r['x']) and np.isfinite(r['y'])])
    vy = np.array([r['y'] for r in test if np.isfinite(r['x']) and np.isfinite(r['y'])])

    bucket_rows = empirical_bucket_stats(train, 20)
    bx = np.array([0.0] + [r['x_mid'] for r in bucket_rows] + [1.0])
    by = np.array([0.0] + [r['median'] for r in bucket_rows] + [1.0])
    weights = np.array([1.0] + [r['count'] for r in bucket_rows] + [1.0])
    order = np.argsort(bx)
    bx, by, weights = bx[order], by[order], weights[order]
    iso_y = pava(np.clip(by, 0, 1), weights)

    def pred_linear(x): return np.clip(x, 0, 1)
    def pred_emp(x): return np.clip(np.interp(x, bx, by), 0, 1)
    def pred_iso(x): return np.clip(np.interp(x, bx, iso_y), 0, 1)

    def logistic_fn(x, k, x0): return 1 / (1 + np.exp(-k * (x - x0)))
    def gompertz_fn(x, a, b): return np.exp(-a * np.exp(-b * x))
    def pow_fn(x, g): return np.clip(x, 1e-9, 1) ** g

    params = {}
    for name, fn, start, bounds in [
        ('Logistic S-curve', logistic_fn, [5.0, 0.5], ([0.05, -2.0], [50.0, 3.0])),
        ('Gompertz curve', gompertz_fn, [5.0, 5.0], ([0.01, 0.01], [100.0, 100.0])),
        ('Power curve', pow_fn, [1.0], ([0.05], [8.0])),
    ]:
        try:
            popt, _ = optimize.curve_fit(fn, tx, ty, p0=start, bounds=bounds, maxfev=20000)
            params[name] = (fn, popt)
        except Exception:
            params[name] = (None, None)

    models = {
        'Linear reference': pred_linear,
        'Empirical median curve': pred_emp,
        'Monotone empirical/isotonic curve': pred_iso,
    }
    for name, (fn, popt) in params.items():
        if fn is not None:
            models[name] = lambda x, fn=fn, popt=popt: np.clip(fn(np.asarray(x), *popt), 0, 1)

    rows = []
    pred_grid = {}
    grid = np.linspace(0, 1, 250)
    for name, fn in models.items():
        pred = np.asarray(fn(vx), dtype=float)
        err = pred - vy
        pred_grid[name] = np.asarray(fn(grid), dtype=float)
        rows.append({
            'ModelFamily': name,
            'MAE': float(np.mean(np.abs(err))) if len(err) else np.nan,
            'RMSE': float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
            'MedianAE': float(np.median(np.abs(err))) if len(err) else np.nan,
            'P90AE': float(np.quantile(np.abs(err), .90)) if len(err) else np.nan,
            'Bias': float(np.mean(err)) if len(err) else np.nan,
        })
    return sorted(rows, key=lambda r: r['MAE']), grid, pred_grid, bx, by, iso_y


def plot_hist(values, title, xlabel, bins=50, logx=False):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if logx:
        arr = arr[arr > 0]
        arrp = np.log10(arr)
        xlabel = 'log10(' + xlabel + ')'
    else:
        arrp = arr
    if len(arrp) == 0:
        arrp = np.array([0])
    lo, hi = np.quantile(arrp, [.01, .99]) if len(arrp) > 5 else (float(np.min(arrp)), float(np.max(arrp)))
    if lo == hi: hi = lo + 1
    pad = (hi - lo) * .08
    lo -= pad; hi += pad
    hist, edges = np.histogram(arrp, bins=np.linspace(lo, hi, bins + 1))
    ymax = max(hist.max(), 1) * 1.15
    w,h=880,470; ml,mr,mt,mb=70,25,34,58
    def sx(x): return ml+(x-lo)/(hi-lo)*(w-ml-mr)
    def sy(y): return h-mb-y/ymax*(h-mt-mb)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
    for frac in [0,.25,.5,.75,1]:
        yy=sy(ymax*frac); parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{w-mr}" y2="{yy:.1f}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{ml-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{int(ymax*frac)}</text>')
    for c,a,b in zip(hist, edges[:-1], edges[1:]):
        x=sx(a); bw=max(1,sx(b)-sx(a)-1); yy=sy(c)
        parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h-mb-yy:.1f}" fill="#60a5fa" opacity="0.65"/>')
    for q,color in [(np.median(arrp),'#111827'),(np.quantile(arrp,.25),'#059669'),(np.quantile(arrp,.75),'#059669')]:
        parts.append(f'<line x1="{sx(q):.1f}" y1="{mt}" x2="{sx(q):.1f}" y2="{h-mb}" stroke="{color}" stroke-width="2" stroke-dasharray="5 5"/>')
    parts.append(f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{w/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">{xml(title)}</text>')
    parts.append(f'<text x="{w/2}" y="{h-10}" text-anchor="middle" font-size="13">{xml(xlabel)}</text></svg>')
    return svg_uri(''.join(parts))


def plot_heatmap(x, y, title):
    arrx=np.asarray([a for a,b in zip(x,y) if np.isfinite(a) and np.isfinite(b)], dtype=float)
    arry=np.asarray([np.clip(b,0,1) for a,b in zip(x,y) if np.isfinite(a) and np.isfinite(b)], dtype=float)
    h,xedges,yedges=np.histogram2d(np.clip(arrx,0,1), arry, bins=[35,35])
    vmax=np.quantile(h[h>0], .95) if np.any(h>0) else 1
    w,hg=900,570; ml,mr,mt,mb=70,80,34,62
    def sx(v): return ml+v*(w-ml-mr)
    def sy(v): return hg-mb-v*(hg-mt-mb)
    def col(c):
        z=min(1,c/vmax) if vmax else 0
        return f'rgb({int(245-190*z)},{int(247-120*z)},{int(250-10*z)})'
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{hg}" viewBox="0 0 {w} {hg}">','<rect width="100%" height="100%" fill="white"/>']
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            c=h[i,j]
            if c<=0: continue
            parts.append(f'<rect x="{sx(xedges[i]):.1f}" y="{sy(yedges[j+1]):.1f}" width="{sx(xedges[i+1])-sx(xedges[i])+.2:.1f}" height="{sy(yedges[j])-sy(yedges[j+1])+.2:.1f}" fill="{col(c)}"/>')
    for t in [0,.25,.5,.75,1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{hg-mb}" stroke="#d1d5db"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{w-mr}" y2="{sy(t):.1f}" stroke="#d1d5db"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{hg-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<line x1="{ml}" y1="{hg-mb}" x2="{w-mr}" y2="{hg-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{hg-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{w/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">{xml(title)}</text>')
    parts.append(f'<text x="{w/2}" y="{hg-10}" text-anchor="middle" font-size="13">Elapsed percent</text><text x="20" y="{hg/2}" transform="rotate(-90 20 {hg/2})" text-anchor="middle" font-size="13">Cumulative project spend percent</text></svg>')
    return svg_uri(''.join(parts))


def plot_percentile_bands(stats_rows, grid, pred_grid):
    w,h=940,570; ml,mr,mt,mb=70,26,34,62
    def sx(x): return ml+x*(w-ml-mr)
    def sy(y): return h-mb-y*(h-mt-mb)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
    for t in [0,.25,.5,.75,1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{h-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{w-mr}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{h-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    xs=[r['x_mid'] for r in stats_rows]
    for lo,hi,color,op in [('p05','p95','#bfdbfe',.60),('p25','p75','#60a5fa',.45)]:
        pts=[(x,r[hi]) for x,r in zip(xs,stats_rows)] + [(x,r[lo]) for x,r in zip(xs[::-1],stats_rows[::-1])]
        parts.append('<polygon points="'+' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in pts)+f'" fill="{color}" opacity="{op}"/>')
    for key,color,lw in [('median','#111827',3),('mean','#f97316',2)]:
        parts.append('<polyline points="'+' '.join(f'{sx(r["x_mid"]):.1f},{sy(r[key]):.1f}' for r in stats_rows)+f'" fill="none" stroke="{color}" stroke-width="{lw}"/>')
    colors={'Linear reference':'#6b7280','Monotone empirical/isotonic curve':'#dc2626','Logistic S-curve':'#7c3aed','Gompertz curve':'#059669','Power curve':'#2563eb'}
    for name,color in colors.items():
        if name in pred_grid:
            dash=' stroke-dasharray="6 5"' if name=='Linear reference' else ''
            parts.append('<polyline points="'+' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(grid,pred_grid[name]))+f'" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
    parts.append(f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{w/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Project cumulative spend bands and candidate curve families</text>')
    parts.append(f'<text x="{w/2}" y="{h-10}" text-anchor="middle" font-size="13">Elapsed percent</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_strata_medians(rows, stratifier, title):
    groups=defaultdict(list)
    for r in rows:
        if np.isfinite(r['x']) and np.isfinite(r['y']): groups[stratifier(r)].append(r)
    w,h=920,540; ml,mr,mt,mb=70,26,34,62
    def sx(x): return ml+x*(w-ml-mr)
    def sy(y): return h-mb-y*(h-mt-mb)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
    for t in [0,.25,.5,.75,1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{h-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{w-mr}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{h-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    colors=['#dc2626','#2563eb','#059669','#f97316','#7c3aed','#111827','#0f766e']
    order=sorted(groups, key=lambda k: {'1 cluster':0,'2 clusters':1,'3 clusters':2,'4-6 clusters':3,'7-12 clusters':4,'13-24 clusters':5,'25+ clusters':6}.get(k,99))
    for i,key in enumerate(order):
        pts=groups[key]
        if len(pts)<20: continue
        stats_rows=empirical_bucket_stats(pts,10)
        color=colors[i%len(colors)]
        parts.append('<polyline points="'+' '.join(f'{sx(r["x_mid"]):.1f},{sy(r["median"]):.1f}' for r in stats_rows)+f'" fill="none" stroke="{color}" stroke-width="3"/>')
        yy=68+i*22
        parts.append(f'<line x1="630" y1="{yy}" x2="660" y2="{yy}" stroke="{color}" stroke-width="3"/><text x="670" y="{yy+4}" font-size="12">{xml(key)} ({len(pts):,})</text>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#6b7280" stroke-width="2" stroke-dasharray="4 5"/>')
    parts.append(f'<line x1="{ml}" y1="{h-mb}" x2="{w-mr}" y2="{h-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{h-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{w/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">{xml(title)}</text><text x="{w/2}" y="{h-10}" text-anchor="middle" font-size="13">Elapsed percent</text></svg>')
    return svg_uri(''.join(parts))


def interpolate_project(project_rows, qs=(.25,.5,.75)):
    xs=[r['x'] for r in project_rows if np.isfinite(r['x']) and np.isfinite(r['y'])]
    ys=[r['y'] for r in project_rows if np.isfinite(r['x']) and np.isfinite(r['y'])]
    if not xs: return None
    order=np.argsort(xs); xs=np.asarray(xs)[order]; ys=np.asarray(ys)[order]
    xs=np.r_[0, xs, 1]; ys=np.r_[0, ys, 1]
    # duplicate x values are possible; keep last cumulative value for interpolation.
    ux=[]; uy=[]
    for x,y in zip(xs,ys):
        if ux and abs(x-ux[-1])<1e-9: uy[-1]=y
        else: ux.append(x); uy.append(y)
    return tuple(float(np.interp(q, ux, uy)) for q in qs)


def main():
    rows, fields = load_rows()
    grouped = group_projects(rows)
    valid = [r for r in rows if np.isfinite(r['x']) and np.isfinite(r['y'])]
    train = [r for r in valid if r['split'] != 'test']
    test = [r for r in valid if r['split'] == 'test']
    projects = list(grouped.keys())

    missing_rows=[]
    for field in fields:
        m=0
        for raw in csv.DictReader(CSV_PATH.open(newline='', encoding='utf-8-sig')):
            if raw.get(field,'')=='': m+=1
        if m: missing_rows.append([field, fmt_int(m), pct(m/max(len(rows),1))])

    customer_counts=Counter(r['customer'] for r in rows)
    customer_projects=Counter(k[0] for k in projects)
    split_rows=Counter(r['split'] for r in rows)
    split_projects=Counter(grouped[k][0]['split'] for k in projects)
    cluster_counts=[len(v) for v in grouped.values()]
    totals=[v[0]['total'] for v in grouped.values() if np.isfinite(v[0]['total']) and v[0]['total']>0]
    durations=[v[0]['days'] for v in grouped.values() if np.isfinite(v[0]['days'])]
    neg_rows=sum(1 for r in rows if np.isfinite(r['cluster_burn']) and r['cluster_burn']<0)
    decreasing=0
    for prs in grouped.values():
        prev=None
        for r in prs:
            if not np.isfinite(r['y_raw']): continue
            if prev is not None and r['y_raw'] + 1e-9 < prev:
                decreasing += 1; break
            prev=r['y_raw']

    overview=[
        ['Input CSV', CSV_PATH.name], ['CSV rows parsed', fmt_int(len(rows))], ['Unique customer/projects', fmt_int(len(projects))],
        ['Customers present', ', '.join(sorted(customer_counts))], ['Train projects', fmt_int(split_projects.get('train',0))], ['Test projects', fmt_int(split_projects.get('test',0))],
        ['Median clusters/project', fmt(np.median(cluster_counts),1)], ['Projects with 1 cluster', f"{fmt_int(sum(c==1 for c in cluster_counts))} ({pct(sum(c==1 for c in cluster_counts)/len(cluster_counts))})"],
        ['Projects with <=6 clusters', f"{fmt_int(sum(c<=6 for c in cluster_counts))} ({pct(sum(c<=6 for c in cluster_counts)/len(cluster_counts))})"],
        ['Negative cluster rows', fmt_int(neg_rows)], ['Projects with decreasing cumulative pct', fmt_int(decreasing)],
    ]
    customer_rows=[[c,fmt_int(customer_counts[c]),fmt_int(customer_projects[c])] for c in sorted(customer_counts)]
    cluster_quant=[[q, fmt(np.quantile(cluster_counts,q),1)] for q in [0,.25,.5,.75,.9,.95,.99,1]]
    distribution_rows=[
        ['Project total burn median', fmt(np.median(totals),2)], ['Project total burn p90', fmt(np.quantile(totals,.9),2)], ['Project total burn p99', fmt(np.quantile(totals,.99),2)],
        ['Project modeled days median', fmt(np.median(durations),1)], ['Project modeled days p90', fmt(np.quantile(durations,.9),1)], ['Project modeled days p99', fmt(np.quantile(durations,.99),1)],
        ['Mean clipped cumulative pct', pct(np.mean([r['y'] for r in valid]))], ['Median clipped cumulative pct', pct(np.median([r['y'] for r in valid]))],
        ['Pearson elapsed vs cumulative pct', fmt(np.corrcoef([r['x'] for r in valid],[r['y'] for r in valid])[0,1],4)], ['Spearman elapsed vs cumulative pct', fmt(stats.spearmanr([r['x'] for r in valid],[r['y'] for r in valid]).correlation,4)],
    ]

    bucket_rows=empirical_bucket_stats(train,20)
    with BUCKET_OUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys())); w.writeheader(); w.writerows(bucket_rows)
    model_rows, grid, pred_grid, bx, by, iso_y = fit_family_models(train,test)
    with MODEL_OUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(model_rows[0].keys())); w.writeheader(); w.writerows(model_rows)
    with PROFILE_OUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['Metric','Value']); w.writerows(overview); w.writerow([]); w.writerow(['Customer','Rows','Projects']); w.writerows(customer_rows); w.writerow([]); w.writerow(['ClusterCountQuantile','Clusters']); w.writerows(cluster_quant)

    # Archetypes by interpolated quartile burn position.
    arche=defaultdict(list)
    for key, prs in grouped.items():
        if len(prs)<2: continue
        vals=interpolate_project(prs)
        if vals is None: continue
        q25,q50,q75=vals
        if q25>=.60: label='front-loaded'
        elif q75<=.60: label='back-loaded'
        elif q50>=.60: label='early-mid weighted'
        elif q50<=.40: label='late-mid weighted'
        else: label='approximately balanced'
        for r in prs: arche[label].append(r)
    arche_rows=[[k, fmt_int(len({(r['customer'],r['project']) for r in v})), fmt_int(len(v))] for k,v in sorted(arche.items())]

    model_table=[[r['ModelFamily'],fmt(r['MAE'],4),fmt(r['RMSE'],4),fmt(r['MedianAE'],4),fmt(r['P90AE'],4),fmt(r['Bias'],4)] for r in model_rows]
    bucket_table=[[r['bucket'],fmt_int(r['count']),pct(r['mean']),pct(r['median']),pct(r['p10']),pct(r['p25']),pct(r['p75']),pct(r['p90']),pct(r['iqr'])] for r in bucket_rows]

    cells=[
        md_cell('# Project-Level Cumulative Spend Exploratory Analysis\n\nThis notebook examines cumulative burn at the project level across all customers. It intentionally does not assume that the project-level curve follows the contract-item-level Beta CDF shape. The goal is to characterize the data, understand sparsity and curve families, and identify reasonable model families for later validation.'),
        md_cell('## Dataset Profile\n\n' + table(['Metric','Value'], overview) + '\n\n### Customer Coverage\n\n' + table(['Customer','Rows','Projects'], customer_rows) + '\n\n### Missing Fields\n\n' + (table(['Field','Missing Rows','Share'], missing_rows) if missing_rows else 'No missing fields detected by empty-string scan.')),
        code_cell(f"import csv\nfrom pathlib import Path\npath = Path({CSV_PATH.name!r})\nwith path.open(newline='', encoding='utf-8-sig') as handle:\n    reader = csv.DictReader(handle)\n    rows = list(reader)\nprint(path)\nprint(len(rows))\nprint(reader.fieldnames)"),
        md_cell('## Project Size, Duration, and Cluster Sparsity\n\nProject-level curves are much more heterogeneous than contract-item curves. Many projects have very few observed payment clusters, which means a single smooth curve family cannot be evaluated fairly without stratifying by observation density.\n\n' + table(['Cluster-count quantile','Clusters'], cluster_quant) + '\n\n<img src="'+plot_hist(cluster_counts,'Clusters per project','clusters per project',bins=60)+'" />\n\n<img src="'+plot_hist(totals,'Project total burn distribution','project total burn',bins=70,logx=True)+'" />\n\n<img src="'+plot_hist(durations,'Project modeled duration distribution','project modeled days',bins=70,logx=True)+'" />'),
        md_cell('## Cumulative Spend Joint Distribution\n\nThe joint density shows the empirical relationship between elapsed project time and cumulative project spend. The diagonal is the pure linear reference. Because many projects have only a few clusters, the density includes many early/final jumps rather than smooth monthly progressions.\n\n' + table(['Metric','Value'], distribution_rows) + '\n\n<img src="'+plot_heatmap([r['x'] for r in valid],[r['y'] for r in valid],'Project elapsed pct vs cumulative spend pct')+'" />'),
        md_cell('## Conditional Percentile Bands and Candidate Curves\n\nThis view is descriptive, not a final model choice. It overlays empirical conditional bands with several candidate curve families: linear, monotone empirical/isotonic, logistic, Gompertz, and power curves. The empirical spread is wide, especially because sparse projects jump directly from low cumulative spend to completion.\n\n' + table(['Elapsed bucket','Rows','Mean','Median','P10','P25','P75','P90','IQR'], bucket_table) + '\n\n<img src="'+plot_percentile_bands(bucket_rows,grid,pred_grid)+'" />'),
        md_cell('## Model Family Screening\n\nThese are preliminary held-out point-level errors, not final production model results. They are useful for screening model families. The main question is whether the project-level shape looks linear, S-shaped, monotone empirical, power-law/front-loaded, or heterogeneous enough to require mixture models.\n\n' + table(['Model family','MAE','RMSE','Median AE','P90 AE','Bias'], model_table)),
        md_cell('## Cluster Count Effects\n\nCluster count is not just a data-quality variable; it changes what curve shapes are observable. One-cluster projects only tell us the final point. Two- and three-cluster projects can look extremely front-loaded or back-loaded depending on when the first large payment occurs. Higher-cluster projects provide enough points to support smoother curve families.\n\n<img src="'+plot_strata_medians(valid, lambda r: bucket_cluster_count(r['clusters']), 'Median project spend curves by project cluster count')+'" />'),
        md_cell('## Duration and Size Effects\n\nDuration and project size can plausibly change curve shape. Short projects may pay in a few bursts; large and long projects are more likely to have smoother cumulative progressions. These strata should be considered before choosing a single global project-level model.\n\n<img src="'+plot_strata_medians(valid, lambda r: bucket_duration(r['days']), 'Median project spend curves by project duration')+'" />\n\n<img src="'+plot_strata_medians(valid, lambda r: bucket_size(r['total']), 'Median project spend curves by project total burn')+'" />'),
        md_cell('## Empirical Curve Archetypes\n\nThe table classifies projects by interpolated spend position at 25%, 50%, and 75% elapsed time. This is a rough descriptive archetype analysis. It suggests whether a mixture/segmented model is more appropriate than a single global curve.\n\n' + table(['Archetype','Projects','Rows'], arche_rows) + '\n\n<img src="'+plot_strata_medians(valid, lambda r: next((k for k,v in arche.items() if r in v), 'unclassified'), 'Median project spend curves by empirical archetype')+'" />'),
        md_cell('## Assessment of Reasonable Model Families\n\nThe project-level distribution does not look like a simple reuse of the contract-item Beta CDF problem. The strongest immediate candidates are:\n\n- **Empirical percentile bands**: best for exploratory reporting and threshold design because they expose the wide conditional spread.\n- **Monotone empirical / isotonic curves**: a conservative global expected-position curve when the goal is calibration rather than parametric elegance.\n- **Piecewise linear curves**: likely practical because project curves are often sparse and payment bursts create segment-like behavior.\n- **Mixture models by cluster count, duration, size, and archetype**: probably necessary if the goal is good project-level prediction across all projects.\n- **Logistic/Gompertz/Richards-type S-curves**: plausible for long, multi-cluster projects, but not for all projects. They should be tested only after excluding or separately handling sparse projects.\n- **Linear baseline**: should remain a benchmark and may be competitive for some strata.\n\nThe immediate next analysis should separate sparse projects from dense projects, then evaluate model families per stratum. A single global parametric curve is unlikely to be adequate for all project-level behavior.'),
        md_cell('## Key Caveats\n\n- This SQL intentionally included all projects with no minimum project spend or cluster-count filter, so the dataset mixes one-cluster, two-cluster, and dense projects.\n- A project is an aggregation of many contract items with different start/end timings; the curve may reflect portfolio composition as much as construction progress.\n- Corrections and negative postings can create non-monotonic raw cumulative percentages, while expected spend models are usually monotone smoothers.\n- Point-level errors overweight projects with many clusters. Project-level validation should also compute one-project-one-weight metrics.\n- The project-level denominator is final historical project spend. Live use will need a clear operational denominator, such as authorized budget or forecast final cost.'),
    ]
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
    OUT_PATH.write_text(json.dumps(nb,indent=2),encoding='utf-8')
    print(f'wrote {OUT_PATH}')
    print(f'wrote {PROFILE_OUT}')
    print(f'wrote {MODEL_OUT}')
    print(f'wrote {BUCKET_OUT}')


if __name__=='__main__':
    main()
