#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

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
OUT_PATH = scoped_path(Path('clustered_curve_proxy_labels.ipynb'))
LABEL_OUT = scoped_path(Path('clustered_curve_proxy_labels.csv'))
DIAG_OUT = scoped_path(Path('clustered_curve_proxy_label_diagnostics.csv'))
LABEL_THRESHOLD = 0.15


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
            })
    return rows


def anchored_design(x, degree):
    x = np.asarray(x, dtype=float)
    cols = [x + 0.0]
    for power in range(degree - 1):
        cols.append(x * (1 - x) * (x ** power))
    return np.vstack(cols).T


def fit_anchored_polynomial(points, degree):
    x = np.array([p['x'] for p in points], dtype=float)
    y = np.clip(np.array([p['y'] for p in points], dtype=float), 0, 1)
    matrix = anchored_design(x, degree)
    coef, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    grid = np.linspace(0, 1, 501)
    raw_grid = anchored_design(grid, degree).dot(coef)
    clipped_grid = np.clip(raw_grid, 0, 1)
    pred = np.clip(matrix.dot(coef), 0, 1)
    errors = pred - y
    return {
        'degree': degree,
        'coef': coef,
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'bias': float(np.mean(errors)),
        'raw_min': float(np.min(raw_grid)),
        'raw_max': float(np.max(raw_grid)),
        'clip_share_grid': float(np.mean((raw_grid < 0) | (raw_grid > 1))),
        'monotonic_violations': int(np.sum(np.diff(clipped_grid) < -1e-6)),
    }


def predict_poly(fit, x):
    matrix = anchored_design(np.asarray(x, dtype=float), fit['degree'])
    return np.clip(matrix.dot(fit['coef']), 0, 1)


def evaluate_fit(fit, points):
    x = np.array([p['x'] for p in points], dtype=float)
    y = np.clip(np.array([p['y'] for p in points], dtype=float), 0, 1)
    pred = predict_poly(fit, x)
    errors = pred - y
    return {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'bias': float(np.mean(errors)),
    }


def select_fit(fits):
    monotone = [f for f in fits if f['monotonic_violations'] == 0 and f['clip_share_grid'] <= 0.01]
    candidates = monotone or sorted(fits, key=lambda f: (f['monotonic_violations'], f['clip_share_grid']))
    return min(candidates, key=lambda f: f['rmse'])


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


def plot_polynomial_fit(rows, fits, selected):
    width, height = 900, 560
    ml, mr, mt, mb = 70, 28, 34, 62

    def sx(x):
        return ml + x * (width - ml - mr)

    def sy(y):
        return height - mb - y * (height - mt - mb)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{width-mr}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')

    pts = rows
    if len(pts) > 2600:
        rng = np.random.default_rng(42)
        pts = [pts[i] for i in rng.choice(len(pts), 2600, replace=False)]
    for r in pts:
        color = '#60a5fa' if r['split'] == 'test' else '#9ca3af'
        opacity = '0.32' if r['split'] == 'test' else '0.16'
        parts.append(f'<circle cx="{sx(r["x"]):.1f}" cy="{sy(np.clip(r["y"], 0, 1)):.1f}" r="1.7" fill="{color}" opacity="{opacity}"/>')

    grid = np.linspace(0, 1, 240)
    colors = {3: '#059669', 4: '#dc2626'}
    for fit in fits:
        pred = predict_poly(fit, grid)
        dash = '' if fit is selected else 'stroke-dasharray="6 5"'
        parts.append('<polyline points="' + ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(grid, pred)) + f'" fill="none" stroke="{colors.get(fit["degree"], "#111827")}" stroke-width="3" {dash}/>')

    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="4 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Polynomial proxy-label reference curves</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13">Elapsed percent</text>')
    parts.append(f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13">Cumulative spend percent</text>')
    legend = [('Train points', '#9ca3af'), ('Test points', '#60a5fa'), ('Degree 3', '#059669'), ('Degree 4', '#dc2626'), ('Linear', '#111827')]
    for i, (label, color) in enumerate(legend):
        y = 72 + i * 24
        if 'points' in label:
            parts.append(f'<circle cx="650" cy="{y-4}" r="5" fill="{color}" opacity="0.45"/>')
        else:
            parts.append(f'<line x1="636" y1="{y-4}" x2="664" y2="{y-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="674" y="{y}" font-size="12">{label}</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_proxy_labels(labeled_rows):
    width, height = 860, 470
    ml, mr, mt, mb = 70, 25, 34, 58
    groups = defaultdict(list)
    for r in labeled_rows:
        if r['split'] != 'test':
            continue
        if r['fast_proxy_label']:
            key = 'Fast proxy'
        elif r['slow_proxy_label']:
            key = 'Slow proxy'
        else:
            key = 'Neutral'
        groups[key].append(r)

    def sx(x):
        return ml + x * (width - ml - mr)

    def sy(y):
        return height - mb - y * (height - mt - mb)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{width-mr}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-34}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    for key, color, opacity in [('Neutral', '#9ca3af', .24), ('Fast proxy', '#dc2626', .55), ('Slow proxy', '#2563eb', .55)]:
        pts = groups[key]
        if len(pts) > 1800:
            rng = np.random.default_rng(42)
            pts = [pts[i] for i in rng.choice(len(pts), 1800, replace=False)]
        for r in pts:
            parts.append(f'<circle cx="{sx(r["elapsed_pct"]):.1f}" cy="{sy(np.clip(r["actual_cumulative_burn_pct"], 0, 1)):.1f}" r="2.0" fill="{color}" opacity="{opacity}"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Held-out proxy labels from polynomial reference</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">Elapsed percent</text>')
    for i, (label, color) in enumerate([('Fast proxy', '#dc2626'), ('Slow proxy', '#2563eb'), ('Neutral', '#9ca3af')]):
        y = 74 + i * 24
        parts.append(f'<circle cx="650" cy="{y-4}" r="5" fill="{color}" opacity="0.65"/><text x="664" y="{y}" font-size="12">{label}</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def build(rows, input_path):
    train = [r for r in rows if r['split'] != 'test']
    test = [r for r in rows if r['split'] == 'test']
    if not test:
        test_items = {(r['customer'], r['item']) for r in rows if abs(hash((r['customer'], r['item']))) % 5 == 0}
        train = [r for r in rows if (r['customer'], r['item']) not in test_items]
        test = [r for r in rows if (r['customer'], r['item']) in test_items]
        for r in train:
            r['split'] = 'train'
        for r in test:
            r['split'] = 'test'

    fits = [fit_anchored_polynomial(train, degree) for degree in (3, 4)]
    selected = select_fit(fits)
    test_eval = {f['degree']: evaluate_fit(f, test) for f in fits}

    labeled = []
    for r in rows:
        expected = float(predict_poly(selected, [r['x']])[0])
        delta = float(np.clip(r['y'], 0, 1) - expected)
        labeled.append({
            'customer': r['customer'],
            'item': r['item'],
            'split': r['split'],
            'cluster_sequence': r['seq'],
            'elapsed_pct': r['x'],
            'actual_cumulative_burn_pct': float(np.clip(r['y'], 0, 1)),
            'proxy_expected_cumulative_burn_pct': expected,
            'proxy_position_delta': delta,
            'proxy_label_threshold': LABEL_THRESHOLD,
            'fast_proxy_label': int(delta > LABEL_THRESHOLD),
            'slow_proxy_label': int(delta < -LABEL_THRESHOLD),
            'proxy_model_degree': selected['degree'],
            'item_total_burn': r['total'],
            'item_cluster_count': r['clusters'],
            'item_modeled_days': r['days'],
        })

    with LABEL_OUT.open('w', newline='', encoding='utf-8') as f:
        fields = list(labeled[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(labeled)

    diag_rows = []
    for fit in fits:
        row = {
            'degree': fit['degree'],
            'selected': int(fit is selected),
            'train_mae': fit['mae'],
            'train_rmse': fit['rmse'],
            'train_bias': fit['bias'],
            'test_mae': test_eval[fit['degree']]['mae'],
            'test_rmse': test_eval[fit['degree']]['rmse'],
            'test_bias': test_eval[fit['degree']]['bias'],
            'raw_min': fit['raw_min'],
            'raw_max': fit['raw_max'],
            'clip_share_grid': fit['clip_share_grid'],
            'monotonic_violations': fit['monotonic_violations'],
            'coefficients': json.dumps([float(c) for c in fit['coef']]),
        }
        diag_rows.append(row)
    with DIAG_OUT.open('w', newline='', encoding='utf-8') as f:
        fields = list(diag_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diag_rows)

    overview = [
        ['Input CSV', input_path.name],
        ['Output labeled CSV', LABEL_OUT.name],
        ['Rows labeled', fmt_int(len(labeled))],
        ['Train rows', fmt_int(len(train))],
        ['Test rows', fmt_int(len(test))],
        ['Selected polynomial degree', selected['degree']],
        ['Label threshold', pct(LABEL_THRESHOLD)],
        ['Fast proxy positives on test', fmt_int(sum(r['fast_proxy_label'] for r in labeled if r['split'] == 'test'))],
        ['Slow proxy positives on test', fmt_int(sum(r['slow_proxy_label'] for r in labeled if r['split'] == 'test'))],
    ]
    diag_table = []
    for fit in fits:
        te = test_eval[fit['degree']]
        diag_table.append([
            fit['degree'],
            'yes' if fit is selected else 'no',
            fmt(fit['mae'], 4),
            fmt(fit['rmse'], 4),
            fmt(te['mae'], 4),
            fmt(te['rmse'], 4),
            fmt(fit['bias'], 4),
            fmt(fit['clip_share_grid'], 4),
            fmt_int(fit['monotonic_violations']),
            ', '.join(fmt(c, 6) for c in fit['coef']),
        ])

    cells = [
        md_cell('# Polynomial Proxy Labels for Spend-Position Evaluation\n\nThis notebook is intentionally separate from the Beta CDF model notebook. It creates retrospective proxy labels from complete historical clustered cumulative spend data and writes `clustered_curve_proxy_labels.csv`. The model notebook consumes that labeled file for ROC/AUC and threshold-sweep analysis.'),
        md_cell('## Method\n\nThe proxy reference is an anchored polynomial fitted on the training rows only. Anchoring forces the curve through `(0, 0)` and `(1, 1)`:\n\n```text\nP(x) = c0*x + x*(1-x)*(c1 + c2*x + ...)\n```\n\nThe fitted curve is clipped to `[0, 1]` before labels are generated. A row is labeled fast when actual cumulative spend is more than the threshold above the polynomial reference, and slow when it is more than the threshold below it.'),
        md_cell('## Output Summary\n\n' + table(['Metric', 'Value'], overview)),
        md_cell('## Polynomial Fit Diagnostics\n\n' + table(['Degree', 'Selected', 'Train MAE', 'Train RMSE', 'Test MAE', 'Test RMSE', 'Train bias', 'Clip share', 'Monotonic violations', 'Coefficients'], diag_table)),
        md_cell('## Reference Curve Plot\n\n<img src="' + plot_polynomial_fit(rows, fits, selected) + '" />'),
        md_cell('## Proxy Label Plot\n\n<img src="' + plot_proxy_labels(labeled) + '" />'),
        code_cell("import csv\nfrom pathlib import Path\n\nlabel_path = Path('clustered_curve_proxy_labels.csv')\nwith label_path.open(newline='', encoding='utf-8') as handle:\n    reader = csv.DictReader(handle)\n    rows = list(reader)\nprint(label_path)\nprint(len(rows))\nprint(reader.fieldnames)\nprint('fast labels:', sum(int(r['fast_proxy_label']) for r in rows if r['split'] == 'test'))\nprint('slow labels:', sum(int(r['slow_proxy_label']) for r in rows if r['split'] == 'test'))"),
        md_cell('## Notes for Downstream Modeling\n\nThese labels are proxy labels, not true budget-overrun or schedule-overrun outcomes. They are still useful because they come from a reference curve fit to complete historical behavior and are separated from the Beta CDF and linear models being evaluated downstream.'),
    ]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    OUT_PATH.write_text(json.dumps(nb, indent=2), encoding='utf-8')


def main():
    path = pick_input()
    rows = load_points(path)
    build(rows, path)
    print(f'wrote {OUT_PATH}')
    print(f'wrote {LABEL_OUT}')
    print(f'wrote {DIAG_OUT}')


if __name__ == '__main__':
    main()
