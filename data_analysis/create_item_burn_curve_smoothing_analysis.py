
#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import uuid
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from scipy.interpolate import PchipInterpolator

IN_PATH = Path('ci_payment_details_2.csv')
CLUSTERS_OUT = Path('ci_item_clustered_cumulative_curves.csv')
SUMMARY_OUT = Path('ci_item_curve_model_summary.csv')
NB_OUT = Path('item_burn_curve_smoothing_analysis.ipynb')

CLUSTER_GAP_DAYS = 3.0


def D(x):
    return float(Decimal(str(x)))


def dt(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')


def fmt(x, p=4):
    if x is None or not np.isfinite(float(x)):
        return ''
    return f'{float(x):,.{p}f}'


def fmt_int(x):
    return f'{int(x):,}'


def pct(x):
    return f'{100*float(x):.2f}%'


def esc(x):
    return str(x).replace('|', '\\|').replace('\n', '<br>')


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for r in rows:
        out.append('| ' + ' | '.join(esc(v) for v in r) + ' |')
    return '\n'.join(out)


def md_cell(source):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': source}


def code_cell(source):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source}


def svg_uri(svg):
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode('ascii')


def load_rows():
    rows = []
    with IN_PATH.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = dict(r)
            row['itemid'] = r['ITEMID']
            row['date'] = dt(r['WPPOSTINGDATE'])
            row['burn'] = D(r['THIS_BURN'])
            row['num_paygroups'] = int(D(r['NUM_PAYGROUPS']))
            row['days_between'] = D(r['DAYS_BETWEEN'])
            rows.append(row)
    return rows


def cluster_rows(rows):
    items = defaultdict(list)
    for r in rows:
        items[r['itemid']].append(r)

    clusters = []
    for itemid, rs in items.items():
        rs.sort(key=lambda r: r['date'])
        current = None
        cluster_seq = 0
        for i, r in enumerate(rs, 1):
            if current is None:
                cluster_seq += 1
                current = {
                    'ITEMID': itemid,
                    'CLUSTER_SEQUENCE': cluster_seq,
                    'CLUSTER_START_DATE': r['date'],
                    'CLUSTER_END_DATE': r['date'],
                    'CLUSTER_BURN': r['burn'],
                    'ROWS_IN_CLUSTER': 1,
                    'NUM_PAYGROUPS': r['num_paygroups'],
                    'DAYS_BETWEEN': r['days_between'],
                }
                prev_date = r['date']
                continue
            gap = (r['date'] - prev_date).total_seconds() / 86400
            if gap <= CLUSTER_GAP_DAYS:
                current['CLUSTER_END_DATE'] = r['date']
                current['CLUSTER_BURN'] += r['burn']
                current['ROWS_IN_CLUSTER'] += 1
            else:
                clusters.append(current)
                cluster_seq += 1
                current = {
                    'ITEMID': itemid,
                    'CLUSTER_SEQUENCE': cluster_seq,
                    'CLUSTER_START_DATE': r['date'],
                    'CLUSTER_END_DATE': r['date'],
                    'CLUSTER_BURN': r['burn'],
                    'ROWS_IN_CLUSTER': 1,
                    'NUM_PAYGROUPS': r['num_paygroups'],
                    'DAYS_BETWEEN': r['days_between'],
                }
            prev_date = r['date']
        if current is not None:
            clusters.append(current)

    by_item = defaultdict(list)
    for c in clusters:
        by_item[c['ITEMID']].append(c)

    enriched = []
    item_summaries = []
    for itemid, cs in by_item.items():
        cs.sort(key=lambda c: c['CLUSTER_END_DATE'])
        first = cs[0]['CLUSTER_END_DATE']
        last = cs[-1]['CLUSTER_END_DATE']
        span_days = max((last - first).total_seconds() / 86400, 0.0)
        # Match the existing convention: first observation receives a 30-day opening period.
        modeled_days = span_days + 30.0
        total_burn = sum(c['CLUSTER_BURN'] for c in cs)
        cumulative = 0.0
        for c in cs:
            cumulative += c['CLUSTER_BURN']
            elapsed_days = (c['CLUSTER_END_DATE'] - first).total_seconds() / 86400
            elapsed_pct = min(max((elapsed_days + 30.0) / modeled_days, 0.0), 1.0) if modeled_days else 1.0
            cum_pct = cumulative / total_burn if abs(total_burn) > 1e-9 else math.nan
            c2 = dict(c)
            c2.update({
                'FIRST_CLUSTER_DATE': first,
                'LAST_CLUSTER_DATE': last,
                'ITEM_TOTAL_BURN': total_burn,
                'ITEM_CLUSTER_COUNT': len(cs),
                'ITEM_SPAN_DAYS': span_days,
                'ITEM_MODELED_DAYS': modeled_days,
                'ELAPSED_DAYS_FROM_FIRST': elapsed_days,
                'ELAPSED_PCT': elapsed_pct,
                'CUMULATIVE_BURN': cumulative,
                'CUMULATIVE_BURN_PCT': cum_pct,
                'TRAIN_SPLIT': 'train' if int(hashlib.md5(itemid.encode()).hexdigest()[:8], 16) % 5 != 0 else 'test',
            })
            enriched.append(c2)
        item_summaries.append({
            'ITEMID': itemid,
            'NUM_PAYMENT_ROWS': sum(c['ROWS_IN_CLUSTER'] for c in cs),
            'NUM_CLUSTERS': len(cs),
            'ITEM_TOTAL_BURN': total_burn,
            'ITEM_SPAN_DAYS': span_days,
            'ITEM_MODELED_DAYS': modeled_days,
            'TRAIN_SPLIT': enriched[-1]['TRAIN_SPLIT'],
        })
    return enriched, item_summaries


def write_clusters(clusters):
    fields = [
        'ITEMID','TRAIN_SPLIT','CLUSTER_SEQUENCE','CLUSTER_START_DATE','CLUSTER_END_DATE','ROWS_IN_CLUSTER','CLUSTER_BURN',
        'CUMULATIVE_BURN','CUMULATIVE_BURN_PCT','ELAPSED_DAYS_FROM_FIRST','ELAPSED_PCT','ITEM_TOTAL_BURN',
        'ITEM_CLUSTER_COUNT','ITEM_SPAN_DAYS','ITEM_MODELED_DAYS','NUM_PAYGROUPS','DAYS_BETWEEN'
    ]
    with CLUSTERS_OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in clusters:
            row = dict(c)
            for d in ['CLUSTER_START_DATE','CLUSTER_END_DATE','FIRST_CLUSTER_DATE','LAST_CLUSTER_DATE']:
                if d in row and hasattr(row[d], 'isoformat'):
                    row[d] = row[d].isoformat(sep=' ')
            writer.writerow({k: row.get(k, '') for k in fields})


def beta_cdf(x, a, b):
    x = np.clip(np.asarray(x, dtype=float), 1e-9, 1 - 1e-9)
    return stats.beta.cdf(x, a, b)


def fit_beta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 1)
    x = x[mask]
    y = np.clip(y[mask], 0, 1)
    def loss(theta):
        a, b = np.exp(theta)
        pred = beta_cdf(x, a, b)
        return np.mean((pred - y) ** 2)
    best = None
    starts = [(1,1),(0.7,1.4),(1.4,0.7),(2,2)]
    for s in starts:
        res = optimize.minimize(loss, np.log(np.array(s)), method='Nelder-Mead', options={'maxiter': 500})
        if best is None or res.fun < best.fun:
            best = res
    a, b = np.exp(best.x)
    return float(a), float(b), float(best.fun)


def empirical_curve(train_points, bins=20):
    x = np.asarray([p[0] for p in train_points])
    y = np.asarray([p[1] for p in train_points])
    centers = []
    medians = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1:
            m = (x >= lo) & (x <= hi)
        else:
            m = (x >= lo) & (x < hi)
        if np.any(m):
            centers.append((lo + hi) / 2)
            medians.append(float(np.median(y[m])))
    xs = np.array([0.0] + centers + [1.0])
    ys = np.array([0.0] + medians + [1.0])
    order = np.argsort(xs)
    xs = xs[order]
    ys = np.maximum.accumulate(np.clip(ys[order], 0, 1))
    return xs, ys


def interp_pred(xs, ys, x):
    return np.interp(np.asarray(x), xs, ys)


def bucket_duration(days):
    if days <= 180: return '<=180d'
    if days <= 365: return '181-365d'
    if days <= 730: return '366-730d'
    return '>730d'


def bucket_clusters(n):
    if n <= 3: return '<=3 clusters'
    if n <= 6: return '4-6 clusters'
    if n <= 12: return '7-12 clusters'
    return '13+ clusters'


def evaluate_models(clusters, item_summaries):
    item_meta = {s['ITEMID']: s for s in item_summaries}
    train_items = {s['ITEMID'] for s in item_summaries if s['TRAIN_SPLIT'] == 'train'}
    test_items = {s['ITEMID'] for s in item_summaries if s['TRAIN_SPLIT'] == 'test'}
    train = [c for c in clusters if c['ITEMID'] in train_items and math.isfinite(c['CUMULATIVE_BURN_PCT'])]
    test = [c for c in clusters if c['ITEMID'] in test_items and math.isfinite(c['CUMULATIVE_BURN_PCT'])]
    train_points = [(c['ELAPSED_PCT'], c['CUMULATIVE_BURN_PCT']) for c in train]

    global_a, global_b, global_loss = fit_beta([p[0] for p in train_points], [p[1] for p in train_points])
    emp_x, emp_y = empirical_curve(train_points, bins=20)

    duration_params = {}
    cluster_params = {}
    for bucket_name, param_store, bucket_func in [
        ('duration', duration_params, lambda s: bucket_duration(s['ITEM_MODELED_DAYS'])),
        ('clusters', cluster_params, lambda s: bucket_clusters(s['NUM_CLUSTERS'])),
    ]:
        grouped = defaultdict(list)
        for c in train:
            grouped[bucket_func(item_meta[c['ITEMID']])].append((c['ELAPSED_PCT'], c['CUMULATIVE_BURN_PCT']))
        for k, pts in grouped.items():
            if len(pts) >= 50:
                param_store[k] = fit_beta([p[0] for p in pts], [p[1] for p in pts])[:2]

    def pred_linear(c): return c['ELAPSED_PCT']
    def pred_global_beta(c): return beta_cdf([c['ELAPSED_PCT']], global_a, global_b)[0]
    def pred_emp(c): return interp_pred(emp_x, emp_y, [c['ELAPSED_PCT']])[0]
    def pred_duration(c):
        k = bucket_duration(item_meta[c['ITEMID']]['ITEM_MODELED_DAYS'])
        a, b = duration_params.get(k, (global_a, global_b))
        return beta_cdf([c['ELAPSED_PCT']], a, b)[0]
    def pred_cluster(c):
        k = bucket_clusters(item_meta[c['ITEMID']]['NUM_CLUSTERS'])
        a, b = cluster_params.get(k, (global_a, global_b))
        return beta_cdf([c['ELAPSED_PCT']], a, b)[0]

    models = {
        'Linear cumulative spend': pred_linear,
        'Global Beta CDF': pred_global_beta,
        'Pooled empirical median curve': pred_emp,
        'Duration-bucket Beta CDF': pred_duration,
        'Cluster-count-bucket Beta CDF': pred_cluster,
    }
    metrics = []
    predictions = defaultdict(dict)
    for name, fn in models.items():
        errs = []
        abs_errs = []
        for c in test:
            y = np.clip(c['CUMULATIVE_BURN_PCT'], 0, 1)
            yhat = float(np.clip(fn(c), 0, 1))
            predictions[name][id(c)] = yhat
            e = yhat - y
            errs.append(e)
            abs_errs.append(abs(e))
        errs = np.array(errs)
        abs_errs = np.array(abs_errs)
        metrics.append({
            'Model': name,
            'MAE': float(np.mean(abs_errs)),
            'RMSE': float(np.sqrt(np.mean(errs ** 2))),
            'MedianAE': float(np.median(abs_errs)),
            'P90AE': float(np.quantile(abs_errs, .90)),
            'Bias': float(np.mean(errs)),
        })

    # Per-item retrospective beta and PCHIP fits: smoothers using the whole completed item history.
    per_item_rows = []
    by_item = defaultdict(list)
    for c in clusters:
        if math.isfinite(c['CUMULATIVE_BURN_PCT']):
            by_item[c['ITEMID']].append(c)
    for itemid, cs in by_item.items():
        if len(cs) < 4 or len(cs) > 80:
            continue
        x = np.array([c['ELAPSED_PCT'] for c in cs])
        y = np.clip(np.array([c['CUMULATIVE_BURN_PCT'] for c in cs]), 0, 1)
        # Use unique x for PCHIP.
        order = np.argsort(x)
        x = x[order]
        y = np.maximum.accumulate(y[order])
        uniq_x, idx = np.unique(x, return_index=True)
        uniq_y = y[idx]
        if len(uniq_x) >= 4:
            try:
                pchip = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
                py = np.clip(pchip(uniq_x), 0, 1)
                pchip_mae = float(np.mean(np.abs(py - uniq_y)))
            except Exception:
                pchip_mae = math.nan
        else:
            pchip_mae = math.nan
        try:
            a, b, _ = fit_beta(x, y)
            by = beta_cdf(x, a, b)
            beta_mae = float(np.mean(np.abs(by - y)))
        except Exception:
            a = b = beta_mae = math.nan
        per_item_rows.append({'ITEMID': itemid, 'N': len(cs), 'BetaA': a, 'BetaB': b, 'BetaMAE': beta_mae, 'PchipMAE': pchip_mae})

    return {
        'train_items': train_items,
        'test_items': test_items,
        'train_points': train,
        'test_points': test,
        'metrics': sorted(metrics, key=lambda m: m['MAE']),
        'global_beta': (global_a, global_b, global_loss),
        'empirical_curve': (emp_x, emp_y),
        'duration_params': duration_params,
        'cluster_params': cluster_params,
        'per_item_rows': per_item_rows,
    }


def rolling_30_day_summary(clusters):
    by_item = defaultdict(list)
    for c in clusters:
        by_item[c['ITEMID']].append(c)
    vals = []
    for itemid, cs in by_item.items():
        cs.sort(key=lambda c: c['CLUSTER_END_DATE'])
        for c in cs:
            end = c['CLUSTER_END_DATE']
            start_ts = end.timestamp() - 30 * 86400
            total = sum(x['CLUSTER_BURN'] for x in cs if start_ts <= x['CLUSTER_END_DATE'].timestamp() <= end.timestamp())
            vals.append(total)
    return vals


def plot_curve(eval_result):
    width, height = 860, 560
    ml, mr, mt, mb = 70, 25, 32, 62
    def sx(x): return ml + x * (width - ml - mr)
    def sy(y): return height - mb - y * (height - mt - mb)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for t in [0,.25,.5,.75,1]:
        x=sx(t); y=sy(t)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height-mb}" stroke="#e5e7eb"/>')
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-35}" text-anchor="middle" font-size="12">{int(t*100)}%</text>')
        parts.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="12">{int(t*100)}%</text>')
    # sample train points lightly
    pts = eval_result['train_points']
    if len(pts) > 2500:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), 2500, replace=False)
        pts = [pts[i] for i in idx]
    for c in pts:
        x, y = c['ELAPSED_PCT'], np.clip(c['CUMULATIVE_BURN_PCT'],0,1)
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.4" fill="#93c5fd" opacity="0.28"/>')
    xs = np.linspace(0,1,220)
    a,b,_ = eval_result['global_beta']
    ys = beta_cdf(xs,a,b)
    parts.append('<polyline points="'+' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(xs,ys))+'" fill="none" stroke="#dc2626" stroke-width="3"/>')
    ex,ey=eval_result['empirical_curve']
    parts.append('<polyline points="'+' '.join(f'{sx(x):.1f},{sy(y):.1f}' for x,y in zip(ex,ey))+'" fill="none" stroke="#059669" stroke-width="3"/>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="6 5"/>')
    parts.append(f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#111827"/><line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#111827"/>')
    parts.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="17" font-weight="700">Clustered cumulative spend curves and fitted expected curves</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13">Elapsed time percent</text>')
    parts.append(f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" font-size="13">Cumulative spend percent</text>')
    lx=585; ly=70
    for i,(label,color,dash) in enumerate([('Train curve points','#93c5fd',''),('Global Beta CDF','#dc2626',''),('Pooled empirical median','#059669',''),('Linear','#111827','dash')]):
        y=ly+i*24
        if i==0: parts.append(f'<circle cx="{lx+13}" cy="{y-4}" r="5" fill="{color}" opacity="0.6"/>')
        else:
            dash_attr = 'stroke-dasharray="6 5"' if dash else ''
            parts.append(f'<line x1="{lx}" y1="{y-4}" x2="{lx+28}" y2="{y-4}" stroke="{color}" stroke-width="3" {dash_attr}/>')
        parts.append(f'<text x="{lx+38}" y="{y}" font-size="12">{label}</text>')
    parts.append('</svg>')
    return svg_uri(''.join(parts))


def main():
    rows = load_rows()
    clusters, item_summaries = cluster_rows(rows)
    write_clusters(clusters)
    eval_result = evaluate_models(clusters, item_summaries)

    with SUMMARY_OUT.open('w', newline='', encoding='utf-8') as f:
        fields = ['Model','MAE','RMSE','MedianAE','P90AE','Bias']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in eval_result['metrics']:
            writer.writerow(m)

    raw_rows = len(rows)
    cluster_count = len(clusters)
    item_count = len(item_summaries)
    rows_per_cluster = raw_rows / cluster_count
    item_cluster_counts = np.array([s['NUM_CLUSTERS'] for s in item_summaries])
    span_days = np.array([s['ITEM_MODELED_DAYS'] for s in item_summaries])
    total_burn = np.array([s['ITEM_TOTAL_BURN'] for s in item_summaries])
    rolling_vals = rolling_30_day_summary(clusters)

    overview_rows = [
        ['Raw payment rows', fmt_int(raw_rows)],
        ['Short-gap cluster threshold', f'{CLUSTER_GAP_DAYS:g} days'],
        ['Clustered burn observations', fmt_int(cluster_count)],
        ['Rows per cluster', fmt(rows_per_cluster, 3)],
        ['Items', fmt_int(item_count)],
        ['Train/test items', f'{fmt_int(len(eval_result["train_items"]))} / {fmt_int(len(eval_result["test_items"]))}'],
        ['Median clusters per item', fmt(np.median(item_cluster_counts), 2)],
        ['P90 clusters per item', fmt(np.quantile(item_cluster_counts, .90), 2)],
        ['Median modeled days per item', fmt(np.median(span_days), 2)],
        ['Median item total burn', fmt(np.median(total_burn), 2)],
    ]
    metric_rows = [[m['Model'], fmt(m['MAE'],4), fmt(m['RMSE'],4), fmt(m['MedianAE'],4), fmt(m['P90AE'],4), fmt(m['Bias'],4)] for m in eval_result['metrics']]
    a,b,loss = eval_result['global_beta']
    beta_rows = [['Global Beta CDF alpha', fmt(a,6)], ['Global Beta CDF beta', fmt(b,6)], ['Train MSE', fmt(loss,8)]]
    dur_rows = [[k, fmt(v[0],6), fmt(v[1],6)] for k,v in sorted(eval_result['duration_params'].items())]
    clus_rows = [[k, fmt(v[0],6), fmt(v[1],6)] for k,v in sorted(eval_result['cluster_params'].items())]
    per_item = eval_result['per_item_rows']
    per_item_beta = [r['BetaMAE'] for r in per_item if np.isfinite(r['BetaMAE'])]
    pchip = [r['PchipMAE'] for r in per_item if np.isfinite(r['PchipMAE'])]
    smoother_rows = [
        ['Per-item Beta CDF median MAE', fmt(np.median(per_item_beta),4)],
        ['Per-item Beta CDF P90 MAE', fmt(np.quantile(per_item_beta,.90),4)],
        ['Per-item monotone PCHIP median interpolation MAE', fmt(np.median(pchip),4)],
    ]
    rolling_rows = [[p, fmt(np.quantile(rolling_vals,p),2)] for p in [.05,.25,.5,.75,.95]]
    plot = plot_curve(eval_result)

    nb = {
        'cells': [
            md_cell('# Item Burn Curve Smoothing Analysis\n\nThis notebook removes payment-row lumpiness by clustering short-gap postings, converting each `ITEMID` to a cumulative spend curve, and evaluating normalized expected-burn curve models.'),
            md_cell('## Method\n\n1. Sort rows by `ITEMID, WPPostingDate`.\n2. Collapse consecutive postings into the same cluster when the gap is `<= 3` days.\n3. Sum `THIS_BURN` within each cluster.\n4. Compute cumulative spend and cumulative spend percent per item.\n5. Compute elapsed percent using the existing convention: `(elapsed_days_from_first_cluster + 30) / (last_cluster_date - first_cluster_date + 30)`.\n6. Fit expected cumulative spend curves on train items and evaluate on held-out items.\n\nThis is retrospective because cumulative percent uses final item total. For production forecasting, the item total scale must be estimated from pre-completion metadata.'),
            code_cell("import csv\nfrom pathlib import Path\n\nwith open('ci_item_clustered_cumulative_curves.csv', newline='', encoding='utf-8') as handle:\n    reader = csv.DictReader(handle)\n    rows = list(reader)\nprint(f'Cluster rows: {len(rows):,}')\nprint(reader.fieldnames)"),
            md_cell('## Data Reduction\n\n' + table(['Metric','Value'], overview_rows)),
            md_cell('## Model Evaluation on Held-Out Items\n\nErrors are in cumulative-spend-percent units. For example, MAE `0.10` means an average absolute error of 10 percentage points of final item spend.\n\n' + table(['Model','MAE','RMSE','Median AE','P90 AE','Bias'], metric_rows)),
            md_cell('## Fitted Curve Parameters\n\n' + table(['Parameter','Value'], beta_rows) + '\n\n### Duration-Bucket Beta Parameters\n\n' + table(['Duration bucket','alpha','beta'], dur_rows) + '\n\n### Cluster-Count-Bucket Beta Parameters\n\n' + table(['Cluster-count bucket','alpha','beta'], clus_rows)),
            md_cell('## Curve Plot\n\nThe points are sampled train clustered cumulative observations. The fitted global Beta curve and pooled empirical median curve are overlaid against a linear reference.\n\n<img src="' + plot + '" />'),
            md_cell('## Retrospective Per-Item Smoothers\n\nThese use each completed item’s full history, so they are smoothers rather than deployable cold-start forecasts. They are useful for estimating smoothed current burn/month after observing an item trajectory.\n\n' + table(['Metric','Value'], smoother_rows)),
            md_cell('## Rolling 30-Day Clustered Burn\n\nA simple operational smoothed burn/month proxy is clustered spend in the trailing 30 days. This is noisy for sparse items but much less lumpy than raw postings.\n\n' + table(['Quantile','Trailing 30-day clustered burn'], rolling_rows)),
            md_cell('## Recommendation\n\nFor the business goal, use cumulative spend curves rather than row-level burn deltas. The best first production shape is a pooled expected cumulative curve plus item-specific updating:\n\n```text\ncluster short-gap postings <= 3 days\nactual_position = cumulative_clustered_spend / estimated_item_total\nexpected_position = expected_curve(elapsed_pct, item_family/duration bucket)\nposition_delta = actual_position - expected_position\nsmoothed_burn_month = item_total_scale * derivative(expected_curve) / modeled_months\n```\n\nWith only the current data, the strongest retrospective baseline is a pooled Beta/empirical cumulative curve. To make it a forward model, add item metadata and active-item snapshots so item total and expected duration can be estimated without using completed-item information.'),
        ],
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}},
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    NB_OUT.write_text(json.dumps(nb, indent=2), encoding='utf-8')
    print('wrote', CLUSTERS_OUT, SUMMARY_OUT, NB_OUT)

if __name__ == '__main__':
    main()
