
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
OUT_PATH = scoped_path(Path('clustered_curve_beta_cdf_model.ipynb'))
SUMMARY_OUT = scoped_path(Path('clustered_curve_beta_cdf_model_summary.csv'))
RISK_OUT = scoped_path(Path('clustered_curve_beta_vs_linear_risk_summary.csv'))
ROC_OUT = scoped_path(Path('clustered_curve_beta_vs_linear_roc_summary.csv'))
PROXY_LABELS_IN = scoped_path(Path('clustered_curve_proxy_labels.csv'))


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



def load_proxy_labels(path):
    labels = {}
    if not path.exists():
        return labels
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            customer = norm_key(r, 'customer', 'CustomerName', 'CUSTOMERNAME') or 'Unknown'
            item = norm_key(r, 'item', 'ITEMID')
            seq = int(float(norm_key(r, 'cluster_sequence', 'ClusterSequence', 'CLUSTER_SEQUENCE') or 0))
            labels[(customer, item, seq)] = {
                'fast': bool(int(float(norm_key(r, 'fast_proxy_label') or 0))),
                'slow': bool(int(float(norm_key(r, 'slow_proxy_label') or 0))),
                'expected': float(norm_key(r, 'proxy_expected_cumulative_burn_pct') or 0),
                'delta': float(norm_key(r, 'proxy_position_delta') or 0),
                'threshold': float(norm_key(r, 'proxy_label_threshold') or 0),
                'degree': int(float(norm_key(r, 'proxy_model_degree') or 0)),
            }
    return labels


def beta_cdf(x, a, b):
    x = np.clip(np.asarray(x, dtype=float), 1e-9, 1 - 1e-9)
    return stats.beta.cdf(x, a, b)


def fit_beta(x, y):
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
    for start in [(1, 1), (.7, 1.4), (1.4, .7), (2, 2), (.45, 1.8), (1.8, .45), (3, 1), (1, 3)]:
        res = optimize.minimize(loss, np.log(np.array(start)), method='Nelder-Mead', options={'maxiter': 1200})
        if best is None or res.fun < best.fun:
            best = res
    a, b = np.exp(best.x)
    return float(a), float(b), float(best.fun)


def empirical_curve(points, bins=20):
    x = np.array([p['x'] for p in points])
    y = np.array([p['y'] for p in points])
    centers, meds = [], []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi if hi == 1 else x < hi)
        if np.any(mask):
            centers.append((lo + hi) / 2)
            meds.append(float(np.median(y[mask])))
    xs = np.array([0.0] + centers + [1.0])
    ys = np.maximum.accumulate(np.clip(np.array([0.0] + meds + [1.0]), 0, 1))
    order = np.argsort(xs)
    return xs[order], ys[order]


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



def roc_curve_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = int(np.sum(labels))
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return np.array([0, 1]), np.array([0, 1]), float('nan')
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    last_score = None
    for score, label in zip(sorted_scores, sorted_labels):
        if last_score is not None and score != last_score:
            tpr.append(tp / pos)
            fpr.append(fp / neg)
        if label:
            tp += 1
        else:
            fp += 1
        last_score = score
    tpr.append(tp / pos)
    fpr.append(fp / neg)
    tpr.append(1.0)
    fpr.append(1.0)
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def threshold_sweep(scores, labels, thresholds):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels))
        tn = int(np.sum(~pred & ~labels))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({'Threshold': threshold, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'Precision': precision, 'RecallTPR': recall, 'FPR': fpr, 'Flagged': int(np.sum(pred))})
    return rows

def evaluate(rows):
    train = [r for r in rows if r['split'] != 'test']
    test = [r for r in rows if r['split'] == 'test']
    if not test:
        test_items = {(r['customer'], r['item']) for r in rows if abs(hash((r['customer'], r['item']))) % 5 == 0}
        train = [r for r in rows if (r['customer'], r['item']) not in test_items]
        test = [r for r in rows if (r['customer'], r['item']) in test_items]

    a, b, train_mse = fit_beta([r['x'] for r in train], [r['y'] for r in train])
    ex, ey = empirical_curve(train)

    dur_params = {}
    clu_params = {}
    for func, store in [(bucket_duration, dur_params), (bucket_clusters, clu_params)]:
        groups = defaultdict(list)
        for r in train:
            key = func(r['days'] if func is bucket_duration else r['clusters'])
            groups[key].append(r)
        for key, pts in groups.items():
            if len(pts) >= 40:
                store[key] = fit_beta([p['x'] for p in pts], [p['y'] for p in pts])[:2]

    def pred_linear(r):
        return r['x']

    def pred_global(r):
        return beta_cdf([r['x']], a, b)[0]

    def pred_emp(r):
        return np.interp(r['x'], ex, ey)

    def pred_dur(r):
        aa, bb = dur_params.get(bucket_duration(r['days']), (a, b))
        return beta_cdf([r['x']], aa, bb)[0]

    def pred_clu(r):
        aa, bb = clu_params.get(bucket_clusters(r['clusters']), (a, b))
        return beta_cdf([r['x']], aa, bb)[0]

    models = {
        'Linear cumulative spend': pred_linear,
        'Global Beta CDF': pred_global,
        'Pooled empirical median curve': pred_emp,
        'Duration-bucket Beta CDF': pred_dur,
        'Cluster-count-bucket Beta CDF': pred_clu,
    }

    metrics = []
    predictions = {name: [] for name in models}
    for name, fn in models.items():
        errs = []
        for r in test:
            yhat = float(np.clip(fn(r), 0, 1))
            predictions[name].append(yhat)
            errs.append(yhat - float(np.clip(r['y'], 0, 1)))
        errs = np.array(errs)
        ae = np.abs(errs)
        metrics.append({
            'Model': name,
            'MAE': float(np.mean(ae)),
            'RMSE': float(np.sqrt(np.mean(errs ** 2))),
            'MedianAE': float(np.median(ae)),
            'P90AE': float(np.quantile(ae, .90)),
            'Bias': float(np.mean(errs)),
        })

    beta_name = 'Duration-bucket Beta CDF'
    linear_name = 'Linear cumulative spend'
    y = np.array([float(np.clip(r['y'], 0, 1)) for r in test])
    pred_beta = np.array(predictions[beta_name])
    pred_linear_arr = np.array(predictions[linear_name])
    residual_beta = y - pred_beta
    residual_linear = y - pred_linear_arr

    risk_rows = []
    thresholds = [0.10, 0.15, 0.25]
    for t in thresholds:
        beta_fast = residual_beta > t
        lin_fast = residual_linear > t
        beta_slow = residual_beta < -t
        lin_slow = residual_linear < -t
        risk_rows.append({
            'Threshold': t,
            'Signal': 'spending too quickly / budget-overrun risk',
            'BetaCount': int(np.sum(beta_fast)),
            'LinearCount': int(np.sum(lin_fast)),
            'BothCount': int(np.sum(beta_fast & lin_fast)),
            'LinearOnly': int(np.sum(lin_fast & ~beta_fast)),
            'BetaOnly': int(np.sum(beta_fast & ~lin_fast)),
        })
        risk_rows.append({
            'Threshold': t,
            'Signal': 'spending too slowly / time-overrun risk',
            'BetaCount': int(np.sum(beta_slow)),
            'LinearCount': int(np.sum(lin_slow)),
            'BothCount': int(np.sum(beta_slow & lin_slow)),
            'LinearOnly': int(np.sum(lin_slow & ~beta_slow)),
            'BetaOnly': int(np.sum(beta_slow & ~lin_slow)),
        })

    proxy_labels = load_proxy_labels(PROXY_LABELS_IN)
    proxy_indices = []
    fast_proxy_label = []
    slow_proxy_label = []
    for i, r in enumerate(test):
        label = proxy_labels.get((r['customer'], r['item'], r['seq']))
        if label is None:
            continue
        proxy_indices.append(i)
        fast_proxy_label.append(label['fast'])
        slow_proxy_label.append(label['slow'])

    fast_proxy_label = np.asarray(fast_proxy_label, dtype=bool)
    slow_proxy_label = np.asarray(slow_proxy_label, dtype=bool)
    proxy_indices = np.asarray(proxy_indices, dtype=int)
    if len(proxy_indices):
        fast_beta_score = residual_beta[proxy_indices]
        fast_linear_score = residual_linear[proxy_indices]
        slow_beta_score = -residual_beta[proxy_indices]
        slow_linear_score = -residual_linear[proxy_indices]

        fpr_fast_beta, tpr_fast_beta, auc_fast_beta = roc_curve_auc(fast_beta_score, fast_proxy_label)
        fpr_fast_linear, tpr_fast_linear, auc_fast_linear = roc_curve_auc(fast_linear_score, fast_proxy_label)
        fpr_slow_beta, tpr_slow_beta, auc_slow_beta = roc_curve_auc(slow_beta_score, slow_proxy_label)
        fpr_slow_linear, tpr_slow_linear, auc_slow_linear = roc_curve_auc(slow_linear_score, slow_proxy_label)
        roc_summary = [
            {'Signal': 'fast_spend_proxy', 'Model': beta_name, 'AUC': auc_fast_beta, 'PositiveLabels': int(np.sum(fast_proxy_label)), 'NegativeLabels': int(len(fast_proxy_label) - np.sum(fast_proxy_label))},
            {'Signal': 'fast_spend_proxy', 'Model': linear_name, 'AUC': auc_fast_linear, 'PositiveLabels': int(np.sum(fast_proxy_label)), 'NegativeLabels': int(len(fast_proxy_label) - np.sum(fast_proxy_label))},
            {'Signal': 'slow_spend_proxy', 'Model': beta_name, 'AUC': auc_slow_beta, 'PositiveLabels': int(np.sum(slow_proxy_label)), 'NegativeLabels': int(len(slow_proxy_label) - np.sum(slow_proxy_label))},
            {'Signal': 'slow_spend_proxy', 'Model': linear_name, 'AUC': auc_slow_linear, 'PositiveLabels': int(np.sum(slow_proxy_label)), 'NegativeLabels': int(len(slow_proxy_label) - np.sum(slow_proxy_label))},
        ]
        sweep_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
        threshold_rows = []
        for signal, model, scores, labels in [
            ('fast_spend_proxy', beta_name, fast_beta_score, fast_proxy_label),
            ('fast_spend_proxy', linear_name, fast_linear_score, fast_proxy_label),
            ('slow_spend_proxy', beta_name, slow_beta_score, slow_proxy_label),
            ('slow_spend_proxy', linear_name, slow_linear_score, slow_proxy_label),
        ]:
            for row in threshold_sweep(scores, labels, sweep_thresholds):
                row = dict(row)
                row['Signal'] = signal
                row['Model'] = model
                threshold_rows.append(row)
    else:
        fpr_fast_beta = tpr_fast_beta = fpr_fast_linear = tpr_fast_linear = np.array([0, 1])
        fpr_slow_beta = tpr_slow_beta = fpr_slow_linear = tpr_slow_linear = np.array([0, 1])
        roc_summary = []
        threshold_rows = []

    return {
        'train': train,
        'test': test,
        'global': (a, b, train_mse),
        'empirical': (ex, ey),
        'duration': dur_params,
        'clusters': clu_params,
        'metrics': sorted(metrics, key=lambda m: m['MAE']),
        'predictions': predictions,
        'residual_beta': residual_beta,
        'residual_linear': residual_linear,
        'risk_rows': risk_rows,
        'roc_summary': roc_summary,
        'threshold_rows': threshold_rows,
        'roc_curves': {
            'fast_beta': (fpr_fast_beta, tpr_fast_beta),
            'fast_linear': (fpr_fast_linear, tpr_fast_linear),
            'slow_beta': (fpr_slow_beta, tpr_slow_beta),
            'slow_linear': (fpr_slow_linear, tpr_slow_linear),
        },
        'proxy_label_rows': int(len(proxy_indices)),
        'proxy_label_file': str(PROXY_LABELS_IN),
        'beta_model_name': beta_name,
        'linear_model_name': linear_name,
    }


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


def plot_curves(result):
    width, height = 900, 570
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

    pts = result['test']
    if len(pts) > 2400:
        rng = np.random.default_rng(42)
        pts = [pts[i] for i in rng.choice(len(pts), 2400, replace=False)]
    for r in pts:
        parts.append(f'<circle cx="{sx(r["x"]):.1f}" cy="{sy(np.clip(r["y"], 0, 1)):.1f}" r="1.8" fill="#60a5fa" opacity="0.30"/>')

    xgrid = np.linspace(0, 1, 240)
    a, b, _ = result['global']
    ybeta = beta_cdf(xgrid, a, b)
    ex, ey = result['empirical']
    parts.append('<polyline points="' + ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(xgrid, ybeta)) + '" fill="none" stroke="#dc2626" stroke-width="3"/>')
    parts.append('<polyline points="' + ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(ex, ey)) + '" fill="none" stroke="#059669" stroke-width="3"/>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="17" font-weight="700">Held-out clustered cumulative spend vs fitted curves</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13">Elapsed percent</text>')
    parts.append(f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13">Cumulative spend percent</text>')
    for i, (label, color) in enumerate([('Held-out cluster points', '#60a5fa'), ('Global Beta CDF', '#dc2626'), ('Pooled empirical median', '#059669'), ('Linear reference', '#111827')]):
        y = 72 + i * 24
        if i == 0:
            parts.append(f'<circle cx="630" cy="{y-4}" r="5" fill="{color}" opacity="0.5"/>')
        else:
            dash = 'stroke-dasharray="6 5"' if i == 3 else ''
            parts.append(f'<line x1="616" y1="{y-4}" x2="644" y2="{y-4}" stroke="{color}" stroke-width="3" {dash}/>')
        parts.append(f'<text x="654" y="{y}" font-size="12">{label}</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_metric_bars(result):
    metrics = {m['Model']: m for m in result['metrics']}
    beta = metrics[result['beta_model_name']]
    linear = metrics[result['linear_model_name']]
    names = ['MAE', 'RMSE', 'P90AE', 'Bias']
    width, height = 820, 440
    ml, mr, mt, mb = 70, 25, 34, 62
    vals = [abs(beta[n]) for n in names] + [abs(linear[n]) for n in names]
    ymax = max(vals) * 1.25

    def sx(i, offset):
        group_w = (width - ml - mr) / len(names)
        return ml + i * group_w + group_w * offset

    def sy(v):
        return height - mb - v / ymax * (height - mt - mb)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for frac in [0, .25, .5, .75, 1]:
        y = sy(ymax * frac)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{ymax*frac:.2f}</text>')
    for i, name in enumerate(names):
        bv, lv = abs(beta[name]), abs(linear[name])
        for off, val, color in [(0.26, bv, '#dc2626'), (0.56, lv, '#111827')]:
            x = sx(i, off)
            bw = 34
            y = sy(val)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{height-mb-y:.1f}" fill="{color}" opacity="0.84"/>')
        parts.append(f'<text x="{sx(i, .50):.1f}" y="{height-35}" text-anchor="middle" font-size="12">{name}</text>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Held-out error comparison: Beta CDF vs linear</text>')
    parts.append('<rect x="575" y="58" width="190" height="58" fill="white" stroke="#d1d5db"/>')
    parts.append('<rect x="592" y="74" width="22" height="12" fill="#dc2626" opacity="0.84"/><text x="624" y="84" font-size="12">Duration-bucket Beta</text>')
    parts.append('<rect x="592" y="98" width="22" height="12" fill="#111827" opacity="0.84"/><text x="624" y="108" font-size="12">Linear</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def plot_residual_hist(result):
    beta = result['residual_beta']
    linear = result['residual_linear']
    bins = np.linspace(-0.8, 0.8, 65)
    bh, edges = np.histogram(beta, bins=bins)
    lh, _ = np.histogram(linear, bins=bins)
    ymax = max(bh.max(), lh.max()) * 1.18
    width, height = 900, 470
    ml, mr, mt, mb = 70, 25, 34, 58

    def sx(x):
        return ml + (x + 0.8) / 1.6 * (width - ml - mr)

    def sy(y):
        return height - mb - y / ymax * (height - mt - mb)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for xt in [-.5, -.25, 0, .25, .5]:
        parts.append(f'<line x1="{sx(xt):.1f}" y1="{mt}" x2="{sx(xt):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(xt):.1f}" y="{height-34}" text-anchor="middle" font-size="12">{xt:+.2f}</text>')
    for frac in [0, .25, .5, .75, 1]:
        y = sy(ymax * frac)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#f3f4f6"/>')
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{int(ymax*frac)}</text>')
    for hist, color, opacity in [(lh, '#111827', .42), (bh, '#dc2626', .50)]:
        pts = []
        for lo, hi, c in zip(edges[:-1], edges[1:], hist):
            mid = (lo + hi) / 2
            pts.append(f'{sx(mid):.1f},{sy(c):.1f}')
        parts.append('<polyline points="' + ' '.join(pts) + f'" fill="none" stroke="{color}" stroke-width="3" opacity="{opacity}"/>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{mt}" x2="{sx(0):.1f}" y2="{height-mb}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Position residual distribution: actual cumulative pct - expected pct</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">Positive = spending faster than expected; negative = spending slower than expected</text>')
    parts.append('<rect x="640" y="58" width="210" height="58" fill="white" stroke="#d1d5db"/>')
    parts.append('<line x1="658" y1="77" x2="688" y2="77" stroke="#dc2626" stroke-width="3" opacity="0.50"/><text x="698" y="81" font-size="12">Duration-bucket Beta</text>')
    parts.append('<line x1="658" y1="101" x2="688" y2="101" stroke="#111827" stroke-width="3" opacity="0.42"/><text x="698" y="105" font-size="12">Linear</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))



def plot_roc_curves(result):
    width, height = 880, 470
    ml, mr, mt, mb = 70, 25, 34, 58
    def sx(x):
        return ml + x * (width - ml - mr)
    def sy(y):
        return height - mb - y * (height - mt - mb)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [0, .25, .5, .75, 1]:
        parts.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{sy(t):.1f}" x2="{width-mr}" y2="{sy(t):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{height-34}" text-anchor="middle" font-size="12">{t:.2f}</text>')
        parts.append(f'<text x="{ml-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="12">{t:.2f}</text>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#9ca3af" stroke-width="2" stroke-dasharray="6 5"/>')
    specs = [
        ('fast_beta', 'Fast: Beta', '#dc2626'),
        ('fast_linear', 'Fast: Linear', '#111827'),
        ('slow_beta', 'Slow: Beta', '#f97316'),
        ('slow_linear', 'Slow: Linear', '#2563eb'),
    ]
    for key, label, color in specs:
        fpr, tpr = result['roc_curves'][key]
        parts.append('<polyline points="' + ' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x, y in zip(fpr, tpr)) + f'" fill="none" stroke="{color}" stroke-width="3" opacity="0.88"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">External proxy-label ROC curves: Beta CDF vs linear position scores</text>')
    parts.append(f'<text x="{width/2}" y="{height-10}" text-anchor="middle" font-size="13">False positive rate</text>')
    parts.append(f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13">True positive rate</text>')
    lx, ly = 610, 70
    for i, (_, label, color) in enumerate(specs):
        y = ly + i * 24
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+30}" y2="{y}" stroke="{color}" stroke-width="3"/><text x="{lx+40}" y="{y+4}" font-size="12">{label}</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))

def build_notebook(input_path, rows, result):
    with SUMMARY_OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'MAE', 'RMSE', 'MedianAE', 'P90AE', 'Bias'])
        writer.writeheader()
        writer.writerows(result['metrics'])
    with RISK_OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Threshold', 'Signal', 'BetaCount', 'LinearCount', 'BothCount', 'LinearOnly', 'BetaOnly'])
        writer.writeheader()
        writer.writerows(result['risk_rows'])
    with ROC_OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Signal', 'Model', 'AUC', 'PositiveLabels', 'NegativeLabels'])
        writer.writeheader()
        writer.writerows(result['roc_summary'])

    item_count = len({(r['customer'], r['item']) for r in rows})
    train_items = len({(r['customer'], r['item']) for r in result['train']})
    test_items = len({(r['customer'], r['item']) for r in result['test']})
    a, b, mse = result['global']
    overview = [
        ['Input CSV', input_path.name],
        ['Cluster curve rows', fmt_int(len(rows))],
        ['Items', fmt_int(item_count)],
        ['Train items', fmt_int(train_items)],
        ['Test items', fmt_int(test_items)],
        ['Global Beta alpha', fmt(a, 6)],
        ['Global Beta beta', fmt(b, 6)],
        ['Global train MSE', fmt(mse, 8)],
        ['Proxy label CSV', result['proxy_label_file']],
        ['Proxy-labeled held-out rows', fmt_int(result['proxy_label_rows'])],
    ]
    metric_rows = [[m['Model'], fmt(m['MAE'], 4), fmt(m['RMSE'], 4), fmt(m['MedianAE'], 4), fmt(m['P90AE'], 4), fmt(m['Bias'], 4)] for m in result['metrics']]
    metric_lookup = {m['Model']: m for m in result['metrics']}
    beta = metric_lookup[result['beta_model_name']]
    linear = metric_lookup[result['linear_model_name']]
    improvement_rows = [
        ['MAE improvement vs linear', pct((linear['MAE'] - beta['MAE']) / linear['MAE'])],
        ['RMSE improvement vs linear', pct((linear['RMSE'] - beta['RMSE']) / linear['RMSE'])],
        ['P90 absolute-error improvement vs linear', pct((linear['P90AE'] - beta['P90AE']) / linear['P90AE'])],
        ['Linear bias', fmt(linear['Bias'], 4)],
        ['Duration-bucket Beta bias', fmt(beta['Bias'], 4)],
    ]
    dur_rows = [[k, fmt(v[0], 6), fmt(v[1], 6)] for k, v in sorted(result['duration'].items())]
    clu_rows = [[k, fmt(v[0], 6), fmt(v[1], 6)] for k, v in sorted(result['clusters'].items())]
    risk_rows = [[fmt(r['Threshold'], 2), r['Signal'], fmt_int(r['BetaCount']), fmt_int(r['LinearCount']), fmt_int(r['BothCount']), fmt_int(r['LinearOnly']), fmt_int(r['BetaOnly'])] for r in result['risk_rows']]
    roc_rows = [[r['Signal'], r['Model'], fmt(r['AUC'], 4), fmt_int(r['PositiveLabels']), fmt_int(r['NegativeLabels'])] for r in result['roc_summary']]
    threshold_rows = [[r['Signal'], r['Model'], fmt(r['Threshold'], 2), fmt_int(r['Flagged']), fmt_int(r['TP']), fmt_int(r['FP']), fmt(r['Precision'], 3), fmt(r['RecallTPR'], 3), fmt(r['FPR'], 3)] for r in result['threshold_rows']]

    cells = [
        md_cell('# Clustered Curve Beta CDF Model\n\nThis notebook consumes the CSV produced by `custpaydetails_clustered_cumulative_curves.sql`, fits Beta CDF expected cumulative burn curves, evaluates held-out items, and compares the Beta CDF approach to a pure linear spend model.'),
        md_cell('## Input and Parameterization\n\n' + table(['Metric', 'Value'], overview)),
        code_cell("import csv\nfrom pathlib import Path\n\npath = Path('clustered_data_input.csv')\nif not path.exists():\n    path = Path('custpaydetails_clustered_cumulative_curves.csv')\nif not path.exists():\n    path = Path('ci_item_clustered_cumulative_curves.csv')\nwith path.open(newline='', encoding='utf-8-sig') as handle:\n    reader = csv.DictReader(handle)\n    rows = list(reader)\nprint(path)\nprint(len(rows))\nprint(reader.fieldnames)"),
        md_cell('## Linear Spend Baseline\n\nThe pure linear model assumes cumulative spend should equal elapsed time:\n\n```text\nexpected_cumulative_pct_linear = elapsed_pct\nposition_delta_linear = actual_cumulative_pct - elapsed_pct\n```\n\nThis is a useful baseline because it is transparent and easy to explain. It is also too rigid: many items are naturally front-loaded or back-loaded, so a linear curve can systematically flag normal burn shapes as risk.'),
        md_cell('## Held-Out Model Performance\n\nErrors are cumulative-spend-percent errors. MAE `0.15` means an average absolute error of about 15 percentage points of final item spend.\n\n' + table(['Model', 'MAE', 'RMSE', 'Median AE', 'P90 AE', 'Bias'], metric_rows)),
        md_cell('## Beta CDF vs Linear: Improvement Summary\n\n' + table(['Comparison', 'Value'], improvement_rows)),
        md_cell('## Error Comparison Plot\n\n<img src="' + plot_metric_bars(result) + '" />'),
        md_cell('## Bucketed Beta CDF Parameters\n\n### Duration Buckets\n\n' + table(['Duration bucket', 'alpha', 'beta'], dur_rows) + '\n\n### Cluster-Count Buckets\n\n' + table(['Cluster-count bucket', 'alpha', 'beta'], clu_rows)),
        md_cell('## Curve Performance Plot\n\n<img src="' + plot_curves(result) + '" />'),
        md_cell('## Risk Signal Framing\n\nFor active items, the same fitted curve can produce budget-overrun and time-overrun risk signals. These are not final labels without a real authorized budget and schedule; they are position signals.\n\n```text\nexpected_pct = model(elapsed_pct)\nposition_delta = actual_cumulative_pct - expected_pct\n\nif position_delta > threshold:\n    spending too quickly / budget-overrun risk\n\nif position_delta < -threshold:\n    spending too slowly / time-overrun risk\n```\n\nFor budget overrun projection once a budget is available:\n\n```text\nprojected_final_spend = actual_cumulative_spend / expected_pct\nprojected_overrun = projected_final_spend - authorized_budget\n```'),
        md_cell('## Beta vs Linear Risk Signals on Held-Out Data\n\nThe table shows how many clustered observations would be flagged by each approach at several position-delta thresholds. `LinearOnly` rows are especially important: these are cases the linear model flags but the duration-bucket Beta curve treats as normal for the observed burn shape.\n\n' + table(['Threshold', 'Signal', 'Beta count', 'Linear count', 'Both', 'Linear only', 'Beta only'], risk_rows)),
        md_cell('## Proxy ROC/AUC Setup\n\nTrue ROC/AUC requires true binary outcome labels. This notebook now consumes retrospective proxy labels from `clustered_curve_proxy_labels.csv`, which is generated by the separate polynomial proxy-label notebook. The Beta CDF and linear models do not create those labels; they only score against them.\n\nThese ROC curves compare how well Beta and linear position scores recover the external proxy labels. They are useful for model behavior comparison, but they are not a substitute for true budget/schedule outcome labels.'),
        md_cell('## Proxy ROC/AUC Summary\n\n' + table(['Signal', 'Model', 'AUC', 'Positive labels', 'Negative labels'], roc_rows)),
        md_cell('## Threshold Sweep for Proxy Labels\n\n' + table(['Signal', 'Model', 'Threshold', 'Flagged', 'TP', 'FP', 'Precision', 'Recall/TPR', 'FPR'], threshold_rows)),
        md_cell('## Proxy ROC Curves\n\n<img src="' + plot_roc_curves(result) + '" />'),
        md_cell('## Residual Distribution Plot\n\nResidual is `actual cumulative pct - expected cumulative pct`. Positive residuals indicate spending ahead of expected position; negative residuals indicate spending behind expected position.\n\n<img src="' + plot_residual_hist(result) + '" />'),
        md_cell('## Recommendations\n\nUse the duration-bucket Beta CDF as the first expected-position model and keep the linear model as a transparent benchmark. For production alerting:\n\n- Use Beta CDF `position_delta` for primary spend-too-fast and spend-too-slow signals.\n- Show the linear delta as a secondary reference because users understand it.\n- Alert only when position delta exceeds a threshold and remains elevated across updates.\n- For budget-overrun detection, combine position delta with authorized budget and projected final spend.\n- For time-overrun detection, combine slow-spend position delta with schedule metadata; slow spending can mean delay, scope removal, or inactive work.'),
    ]
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    OUT_PATH.write_text(json.dumps(nb, indent=2), encoding='utf-8')


def main():
    path = pick_input()
    rows = load_points(path)
    result = evaluate(rows)
    build_notebook(path, rows, result)
    print(f'wrote {OUT_PATH} using {path}')
    print(f'wrote {SUMMARY_OUT}')
    print(f'wrote {RISK_OUT}')
    print(f'wrote {ROC_OUT}')


if __name__ == '__main__':
    main()
