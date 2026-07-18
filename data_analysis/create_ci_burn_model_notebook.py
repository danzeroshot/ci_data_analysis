
#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median

CSV_PATH = Path('ci_payment_details_2.csv')
OUT_PATH = Path('ci_burn_model_analysis.ipynb')


def D(value):
    if value is None or str(value).strip() == '':
        return None
    try:
        return float(Decimal(str(value).strip()))
    except InvalidOperation:
        return None


def dt(value):
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f')


def fmt_int(x):
    return f'{int(x):,}'


def fmt_num(x, places=2):
    if x is None:
        return ''
    return f'{float(x):,.{places}f}'


def fmt_pct(x):
    return f'{float(x):.2%}'


def esc(x):
    return str(x).replace('|', '\\|').replace('\n', '<br>')


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(esc(x) for x in row) + ' |')
    return '\n'.join(out)


def q(vals, p):
    vals = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not vals:
        return None
    return vals[int((len(vals) - 1) * p)]


def bucket(value, cuts):
    for i, cut in enumerate(cuts):
        if value <= cut:
            return i
    return len(cuts)


def bucket_label(i, cuts):
    edges = [0] + cuts + [1]
    if i == 0:
        return f'<= {cuts[0]:.0%}'
    if i == len(cuts):
        return f'> {cuts[-1]:.0%}'
    return f'{cuts[i-1]:.0%}-{cuts[i]:.0%}'


def md_cell(source):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': source}


def code_cell(source):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}


def load_rows():
    rows = []
    with CSV_PATH.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = dict(row)
            row['itemid'] = row['ITEMID']
            row['date'] = dt(row['WPPOSTINGDATE'])
            row['first'] = dt(row['FIRSTPOSTINGDATE'])
            row['last'] = dt(row['LASTPOSTINGDATE'])
            row['num_paygroups'] = int(D(row['NUM_PAYGROUPS']))
            row['days_between'] = int(D(row['DAYS_BETWEEN']))
            row['base_monthly'] = D(row['CI_BURN_RATE_MONTHS'])
            row['burn'] = D(row['THIS_BURN'])
            row['delta_pct'] = D(row['BURN_DELTA_MONTHS_PERCENT'])
            row['delta_abs'] = D(row['BURN_DELTA_MONTHS'])
            row['ratio_to_monthly'] = row['burn'] / row['base_monthly'] if row['base_monthly'] else None
            row['train'] = int(hashlib.md5(row['itemid'].encode()).hexdigest()[:8], 16) % 5 != 0
            rows.append(row)
    return rows


def enrich(rows):
    items = defaultdict(list)
    for row in rows:
        items[row['itemid']].append(row)
    for itemid, item_rows in items.items():
        item_rows.sort(key=lambda r: r['date'])
        prev = None
        n = len(item_rows)
        item_total_burn = sum(r['burn'] for r in item_rows)
        modeled_months = (item_rows[0]['days_between'] + 30) / 30
        for i, row in enumerate(item_rows, 1):
            row['seq'] = i
            row['seq_frac'] = (i - 1) / (n - 1) if n > 1 else 0
            span = (row['last'] - row['first']).total_seconds() / 86400
            row['time_frac'] = ((row['date'] - row['first']).total_seconds() / 86400) / span if span else 0
            row['interval_days'] = 30.0 if i == 1 else (row['date'] - prev).total_seconds() / 86400
            row['interval_months'] = row['interval_days'] / 30
            row['interval_linear_pred'] = row['base_monthly'] * row['interval_months']
            row['item_total_burn'] = item_total_burn
            row['modeled_months'] = modeled_months
            prev = row['date']
    return items


def metrics(data, pred):
    errors = []
    abs_errors = []
    sq_errors = []
    scaled_by_base = []
    ape_nonzero = []
    for row in data:
        y = row['burn']
        yhat = pred(row)
        e = y - yhat
        errors.append(e)
        abs_errors.append(abs(e))
        sq_errors.append(e * e)
        scaled_by_base.append(abs(e) / row['base_monthly'])
        if abs(y) > 1e-9:
            ape_nonzero.append(abs(e) / abs(y))
    return {
        'bias': mean(errors),
        'mae': mean(abs_errors),
        'rmse': math.sqrt(mean(sq_errors)),
        'median_ae': median(abs_errors),
        'mean_abs_err_over_monthly_rate': mean(scaled_by_base),
        'median_abs_err_over_monthly_rate': median(scaled_by_base),
        'mape_nonzero_actuals': mean(ape_nonzero),
    }


def analyze():
    rows = load_rows()
    items = enrich(rows)
    train = [r for r in rows if r['train']]
    test = [r for r in rows if not r['train']]
    train_items = {r['itemid'] for r in train}
    test_items = {r['itemid'] for r in test}

    seq_cuts = [.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95]
    time_cuts = seq_cuts
    num_cuts = [8, 12, 20, 40, 80]

    med_ratio = median(r['ratio_to_monthly'] for r in train)
    mean_ratio = mean(r['ratio_to_monthly'] for r in train)

    seq_ratios = defaultdict(list)
    time_ratios = defaultdict(list)
    combo_ratios = defaultdict(list)
    interval_ratios = defaultdict(list)
    for row in train:
        sb = bucket(row['seq_frac'], seq_cuts)
        tb = bucket(row['time_frac'], time_cuts)
        nb = bucket(row['num_paygroups'], num_cuts)
        seq_ratios[sb].append(row['ratio_to_monthly'])
        time_ratios[tb].append(row['ratio_to_monthly'])
        combo_ratios[(sb, nb)].append(row['ratio_to_monthly'])
        if abs(row['interval_linear_pred']) > 1e-9:
            interval_ratios[sb].append(row['burn'] / row['interval_linear_pred'])

    seq_factor = {k: median(v) for k, v in seq_ratios.items()}
    time_factor = {k: median(v) for k, v in time_ratios.items()}
    combo_factor = {k: median(v) for k, v in combo_ratios.items() if len(v) >= 30}
    interval_seq_factor = {k: median(v) for k, v in interval_ratios.items() if len(v) >= 30}

    def seq_bucket(row):
        return bucket(row['seq_frac'], seq_cuts)

    def time_bucket(row):
        return bucket(row['time_frac'], time_cuts)

    def num_bucket(row):
        return bucket(row['num_paygroups'], num_cuts)

    models = {
        'Current monthly-rate per row': lambda r: r['base_monthly'],
        'Global median row/month ratio': lambda r: r['base_monthly'] * med_ratio,
        'Global mean row/month ratio': lambda r: r['base_monthly'] * mean_ratio,
        'Sequence-bucket median ratio': lambda r: r['base_monthly'] * seq_factor.get(seq_bucket(r), med_ratio),
        'Calendar-progress median ratio': lambda r: r['base_monthly'] * time_factor.get(time_bucket(r), med_ratio),
        'Sequence x item-size median ratio': lambda r: r['base_monthly'] * combo_factor.get((seq_bucket(r), num_bucket(r)), seq_factor.get(seq_bucket(r), med_ratio)),
        'Interval-weighted linear allocation': lambda r: r['interval_linear_pred'],
        'Interval allocation x seq residual median': lambda r: r['interval_linear_pred'] * interval_seq_factor.get(seq_bucket(r), 1.0),
    }
    metric_rows = []
    metric_values = {}
    for name, pred in models.items():
        m = metrics(test, pred)
        metric_values[name] = m
        metric_rows.append([
            name,
            fmt_num(m['bias'], 2),
            fmt_num(m['mae'], 2),
            fmt_num(m['rmse'], 2),
            fmt_num(m['median_ae'], 2),
            fmt_num(m['mean_abs_err_over_monthly_rate'], 4),
            fmt_num(m['median_abs_err_over_monthly_rate'], 4),
            fmt_num(m['mape_nonzero_actuals'], 4),
        ])

    ratio_vals = [r['ratio_to_monthly'] for r in rows]
    delta_vals = [r['delta_pct'] for r in rows]
    burn_vals = [r['burn'] for r in rows]
    interval_ratio_vals = [r['burn'] / r['interval_linear_pred'] for r in rows if abs(r['interval_linear_pred']) > 1e-9]

    summary_rows = [
        ['Rows', fmt_int(len(rows))],
        ['Items', fmt_int(len(items))],
        ['Train items / rows', f'{fmt_int(len(train_items))} / {fmt_int(len(train))}'],
        ['Test items / rows', f'{fmt_int(len(test_items))} / {fmt_int(len(test))}'],
        ['Median THIS_BURN', fmt_num(q(burn_vals, .50), 2)],
        ['Median monthly baseline', fmt_num(q([r['base_monthly'] for r in rows], .50), 2)],
        ['Median actual/monthly-rate ratio', fmt_num(q(ratio_vals, .50), 4)],
        ['Mean actual/monthly-rate ratio', fmt_num(mean(ratio_vals), 4)],
        ['Rows within +/-25% of monthly baseline', f'{fmt_int(sum(1 for v in delta_vals if abs(v) <= 25))} ({fmt_pct(sum(1 for v in delta_vals if abs(v) <= 25) / len(rows))})'],
        ['Rows below -100% delta', fmt_int(sum(1 for v in delta_vals if v < -100))],
        ['Rows above +100% delta', fmt_int(sum(1 for v in delta_vals if v > 100))],
        ['Negative burn rows', fmt_int(sum(1 for v in burn_vals if v < 0))],
    ]

    seq_rows = []
    for k in sorted(seq_ratios):
        all_bucket = [r for r in rows if seq_bucket(r) == k]
        vals = [r['ratio_to_monthly'] for r in all_bucket]
        deltas = [r['delta_pct'] for r in all_bucket]
        seq_rows.append([
            bucket_label(k, seq_cuts),
            fmt_int(len(all_bucket)),
            fmt_num(q(vals, .25), 4),
            fmt_num(q(vals, .50), 4),
            fmt_num(q(vals, .75), 4),
            fmt_num(q(deltas, .50), 2),
            fmt_pct(sum(1 for r in all_bucket if r['burn'] < 0) / len(all_bucket)),
        ])

    interval_rows = []
    for label, filt in [
        ('First payment', lambda r: r['seq'] == 1),
        ('Middle payments', lambda r: 1 < r['seq'] < r['num_paygroups']),
        ('Last payment', lambda r: r['seq'] == r['num_paygroups']),
    ]:
        subset = [r for r in rows if filt(r)]
        ratios = [r['burn'] / r['interval_linear_pred'] for r in subset if abs(r['interval_linear_pred']) > 1e-9]
        interval_rows.append([
            label,
            fmt_int(len(subset)),
            fmt_num(q([r['interval_days'] for r in subset], .50), 2),
            fmt_num(q(ratios, .25), 4),
            fmt_num(q(ratios, .50), 4),
            fmt_num(q(ratios, .75), 4),
            fmt_num(q(ratios, .95), 4),
        ])

    item_errors = []
    for itemid, item_rows in items.items():
        item_errors.append(sum(r['burn'] for r in item_rows) - sum(r['interval_linear_pred'] for r in item_rows))
    total_recon_rows = [
        ['Median absolute item total error', fmt_num(median(abs(v) for v in item_errors), 6)],
        ['Max absolute item total error', fmt_num(max(abs(v) for v in item_errors), 6)],
        ['Items within $1 of item total', fmt_int(sum(1 for v in item_errors if abs(v) <= 1))],
        ['Items within $100 of item total', fmt_int(sum(1 for v in item_errors if abs(v) <= 100))],
    ]

    outliers_pos = sorted(rows, key=lambda r: r['delta_pct'], reverse=True)[:15]
    outliers_neg = sorted(rows, key=lambda r: r['delta_pct'])[:15]
    outlier_headers = ['ITEMID', 'Seq', 'Num', 'Date', 'Interval days', 'Monthly base', 'This burn', 'Delta %']
    def outlier_row(r):
        return [r['itemid'], r['seq'], r['num_paygroups'], r['WPPOSTINGDATE'], fmt_num(r['interval_days'], 2), fmt_num(r['base_monthly'], 2), fmt_num(r['burn'], 2), fmt_num(r['delta_pct'], 2)]

    item_cv_rows = []
    for itemid, item_rows in items.items():
        vals = [r['ratio_to_monthly'] for r in item_rows]
        avg = mean(vals)
        sd = math.sqrt(mean((v - avg) ** 2 for v in vals)) if vals else 0
        item_cv_rows.append([itemid, len(item_rows), fmt_num(sum(r['burn'] for r in item_rows), 2), fmt_num(avg, 4), fmt_num(sd, 4), fmt_num(max(vals), 4), fmt_int(sum(1 for r in item_rows if r['burn'] < 0))])
    item_cv_rows = sorted(item_cv_rows, key=lambda row: float(row[4].replace(',', '')), reverse=True)[:20]

    conclusion = (
        'The best model depends on the job. For allocating a known item total across the item timeline, the interval-weighted model is structurally best because it conserves item totals. '
        'For predicting the typical row, a robust multiplicative model on top of `CI_BURN_RATE_MONTHS` performs better on median error because payment events are bursty and payment dates do not behave like pure accrual boundaries. '
        'A practical model should therefore be hierarchical and probabilistic: use item-level scale, model row-level share/ratio with sequence and timing features, and return prediction intervals rather than a single deterministic burn.'
    )

    cells = [
        md_cell('# Burn Model Analysis for `ci_payment_details_2.csv`\n\nThis notebook analyzes what a good burn model should look like for the revised payment-detail data. It focuses on the `_months` fields and treats each row as one payment event for one contract item.'),
        md_cell('## Core Conclusion\n\n' + conclusion),
        md_cell('## Dataset and Modeling Setup\n\n' + md_table(['Metric', 'Value'], summary_rows) + '\n\nRows are split by `ITEMID` into train/test groups using a deterministic hash. This prevents rows from the same item appearing in both train and test. That matters because item-level scale is repeated across every row for an item.'),
        code_cell("import csv\nfrom pathlib import Path\n\nCSV_PATH = Path('ci_payment_details_2.csv')\nwith CSV_PATH.open(newline='', encoding='utf-8-sig') as handle:\n    reader = csv.DictReader(handle)\n    rows = list(reader)\n\nprint(f'Rows: {len(rows):,}')\nprint(f'Fields: {reader.fieldnames}')"),
        md_cell('## Important Leakage / Scope Caveat\n\n`CI_BURN_RATE_MONTHS` appears to be derived from the item total burn divided by modeled months. That makes it an excellent normalization field for retrospective analysis, but it can leak future information if the task is to forecast an item before its total burn is known.\n\nSo there are two different modeling problems:\n\n1. **Retrospective allocation / anomaly detection:** Given item scale and duration, estimate whether each payment event is unusually high or low. This file supports that well.\n2. **Forward forecasting before item completion:** Estimate future burn without knowing final item total. This file alone is not enough; it needs item quantity, unit price, item type, contract, project, and schedule covariates.'),
        md_cell('## Why the Current Monthly Delta Is Not a Full Model\n\n`BURN_DELTA_MONTHS_PERCENT` compares each payment event to a full monthly burn rate. But each row is a payment event, not a guaranteed one-month period. The median row is only about 22% of the monthly baseline, so most rows look negative against a full-month comparator. That does not automatically mean underperformance; it often means the row represents a smaller accounting/payment slice.'),
        md_cell('## Model Comparison on Held-Out Items\n\n' + md_table(['Model', 'Bias', 'MAE', 'RMSE', 'Median AE', 'Mean abs err / monthly rate', 'Median abs err / monthly rate', 'MAPE nonzero actuals'], metric_rows)),
        md_cell('## Interpreting the Model Comparison\n\n- The current monthly-rate-per-row baseline is systematically too high for most individual rows. It has a large negative bias because predicted burn is one full month of burn for every payment event.\n- The global median row/month ratio has low median absolute error because it predicts a typical payment event as a fraction of monthly burn, not a full month.\n- The global mean ratio is less biased in dollars, but worse for the median row because the positive tail is large.\n- Sequence and calendar-progress median-ratio models make only modest row-level improvements. That means the dominant signal is item scale plus heavy-tailed payment-event noise, not a smooth deterministic curve.\n- The interval-weighted linear allocation is the best structural allocation model: it nearly reconciles item totals, but row-level errors remain large because payment dates are not clean work-accrual period boundaries.'),
        md_cell('## Sequence Shape\n\nThis table describes actual burn as a ratio of the monthly baseline by payment-event position. A value of `0.25` means the median payment in that bucket is one quarter of the item monthly burn rate.\n\n' + md_table(['Sequence fraction bucket', 'Rows', 'P25 actual/monthly', 'Median actual/monthly', 'P75 actual/monthly', 'Median delta %', 'Negative burn share'], seq_rows)),
        md_cell('## Calendar-Interval Allocation\n\nA natural linear model is: `predicted burn = CI_BURN_RATE_MONTHS * interval_days / 30`, where the first event receives 30 days and later events receive days since prior payment. This model aligns with the `(DAYS_BETWEEN + 30) / 30` construction.\n\n' + md_table(['Payment position', 'Rows', 'Median interval days', 'P25 actual/interval pred', 'Median actual/interval pred', 'P75 actual/interval pred', 'P95 actual/interval pred'], interval_rows)),
        md_cell('## Item-Total Reconciliation of Interval Model\n\n' + md_table(['Metric', 'Value'], total_recon_rows) + '\n\nThis is why the interval-weighted model is useful even when it is not the best row-level point predictor. It allocates the known item total across time in a way that is internally coherent.'),
        md_cell('## Extreme Positive Rows\n\n' + md_table(outlier_headers, [outlier_row(r) for r in outliers_pos])),
        md_cell('## Extreme Negative Rows\n\nRows below `-100%` delta are driven by negative actual burn. These should be modeled as corrections/reversals or flagged separately.\n\n' + md_table(outlier_headers, [outlier_row(r) for r in outliers_neg])),
        md_cell('## Most Volatile Items by Row/Monthly Ratio\n\n' + md_table(['ITEMID', 'Rows', 'Total burn', 'Mean actual/monthly', 'Std dev actual/monthly', 'Max actual/monthly', 'Negative rows'], item_cv_rows)),
        md_cell('## Recommended Model\n\nA good model for this data should be **hierarchical, scale-aware, and probabilistic**.\n\nRecommended structure:\n\n1. **Item scale layer:** Estimate or accept an item-level total burn and duration. In this retrospective file, `CI_BURN_RATE_MONTHS` supplies that scale. In a forward model, estimate scale from contract item quantity, unit price, item family, project/contract metadata, and schedule information.\n2. **Exposure layer:** Convert payment timing into exposure. The simplest exposure is `interval_days / 30`, with first event exposure set to one month. This gives a total-conserving allocation.\n3. **Event-shape layer:** Add a multiplicative factor for payment sequence, calendar progress, item duration, and number of paygroups. Use robust medians or a regularized model because the distribution is heavy-tailed.\n4. **Correction layer:** Treat negative burns separately with a classifier or flag. They are real behavior but should not be forced into the same continuous positive-burn distribution.\n5. **Uncertainty layer:** Predict quantiles or intervals, not just means. The tail is too large for a single point estimate to be honest.'),
        md_cell("""## Practical Formula

For retrospective allocation/anomaly detection:

```text
base_monthly = CI_BURN_RATE_MONTHS
exposure_months = 1.0 for first payment, else days_since_prior_payment / 30
structural_pred = base_monthly * exposure_months
row_pred = structural_pred * sequence_residual_factor
residual = THIS_BURN - row_pred
```

For robust row-level expected payment size when conservation is less important:

```text
row_pred = CI_BURN_RATE_MONTHS * median_ratio(sequence_bucket, item_size_bucket)
```

Use the first formula when item-total reconciliation matters. Use the second when the goal is the typical payment-event amount. For forecasting, replace `CI_BURN_RATE_MONTHS` with a separately estimated item scale to avoid leaking completed-item totals."""),
        md_cell('## Bottom Line\n\nThe burn process is not well described by evenly spreading monthly burn across payment rows. Payment rows are bursty accounting events. The strongest defensible model is a two-level model: estimate item-level scale, then allocate or predict payment-event burn as a noisy, heavy-tailed share of that scale using exposure and sequence features. Negative burns should be explicit correction events, not ordinary low burns.'),
    ]

    nb = {
        'cells': cells,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}},
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    OUT_PATH.write_text(json.dumps(nb, indent=2), encoding='utf-8')
    print(f'Wrote {OUT_PATH} with {len(cells)} cells')


if __name__ == '__main__':
    analyze()
