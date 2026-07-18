
#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median

CSV_PATH = Path('ci_payment_details_2.csv')
OUT_PATH = Path('ci_payment_details_2_eda.ipynb')

DATE_FIELDS = ['WPPOSTINGDATE', 'FIRSTPOSTINGDATE', 'LASTPOSTINGDATE']
NUMERIC_FIELDS = [
    'ITEMID', 'NUM_PAYGROUPS', 'DAYS_BETWEEN', 'CI_BURN_RATE_MONTHS', 'THIS_BURN',
    'BURN_DELTA_MONTHS_PERCENT', 'BURN_DELTA_MONTHS', 'CI_BURN_RATE_PG',
    'BURN_DELTA_PG_PERCENT', 'BURN_DELTA_PG'
]
MONTH_FIELDS = ['CI_BURN_RATE_MONTHS', 'BURN_DELTA_MONTHS_PERCENT', 'BURN_DELTA_MONTHS']
PG_FIELDS = ['CI_BURN_RATE_PG', 'BURN_DELTA_PG_PERCENT', 'BURN_DELTA_PG']

FIELD_NOTES = {
    'ITEMID': 'Contract item identifier. User described each ITEMID as a particular contract item with multiple payment postings.',
    'WPPOSTINGDATE': 'Posting date for this actual burn observation.',
    'FIRSTPOSTINGDATE': 'First posting date observed for the ITEMID.',
    'LASTPOSTINGDATE': 'Last posting date observed for the ITEMID.',
    'NUM_PAYGROUPS': 'Total number of postings/pay groups for the ITEMID according to the query.',
    'DAYS_BETWEEN': 'Days between first and last posting for the ITEMID.',
    'CI_BURN_RATE_PG': 'Linear item burn rate using fixed pay-group units. Ignored for primary interpretation in this notebook.',
    'CI_BURN_RATE_MONTHS': 'Linear item burn rate using elapsed months: roughly item total burn / ((DAYS_BETWEEN + 30) / 30).',
    'THIS_BURN': 'Actual burn amount/rate for this row.',
    'BURN_DELTA_MONTHS_PERCENT': 'Monthly-model normalized deviation: (THIS_BURN - CI_BURN_RATE_MONTHS) / CI_BURN_RATE_MONTHS * 100.',
    'BURN_DELTA_PG_PERCENT': 'Pay-group-model normalized deviation. Ignored for primary interpretation in this notebook.',
    'BURN_DELTA_PG': 'Absolute delta from pay-group linear model. Ignored for primary interpretation in this notebook.',
    'BURN_DELTA_MONTHS': 'Absolute delta from monthly linear model: THIS_BURN - CI_BURN_RATE_MONTHS.'
}


def D(value):
    if value is None or str(value).strip() == '':
        return None
    try:
        return Decimal(str(value).strip())
    except InvalidOperation:
        return None


def parse_dt(value):
    if value is None or str(value).strip() == '':
        return None
    text = str(value).strip()
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def fmt_int(x):
    return f'{int(x):,}'


def fmt_pct(x):
    return f'{float(x):.2%}'


def fmt_dec(x, places=2):
    if x is None:
        return ''
    return f'{float(x):,.{places}f}'


def md_escape(x):
    text = '' if x is None else str(x)
    return text.replace('|', '\\|').replace('\n', '<br>')


def table(headers, rows, limit=None):
    shown = rows if limit is None else rows[:limit]
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in shown:
        out.append('| ' + ' | '.join(md_escape(x) for x in row) + ' |')
    if limit is not None and len(rows) > limit:
        out.append('| ' + f'... {len(rows)-limit:,} more rows' + ' |' + ' |'.join([''] * (len(headers)-1)) + ' |')
    return '\n'.join(out)


def q(values, p):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = Decimal(str(p)) * Decimal(len(vals) - 1)
    lo = int(pos // 1)
    hi = int(math.ceil(float(pos)))
    if lo == hi:
        return vals[lo]
    frac = pos - Decimal(lo)
    return vals[lo] + (vals[hi] - vals[lo]) * frac


def code_cell(source):
    return {'cell_type': 'code', 'id': uuid.uuid4().hex[:8], 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}


def md_cell(source):
    return {'cell_type': 'markdown', 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': source}


def bucket_num_paygroups(n):
    if n <= 8: return '07-08'
    if n <= 12: return '09-12'
    if n <= 18: return '13-18'
    if n <= 30: return '19-30'
    return '31+'


def bucket_days(days):
    if days < 180: return '<180'
    if days < 365: return '180-364'
    if days < 730: return '365-729'
    if days < 1095: return '730-1094'
    return '1095+'


def bucket_pct(pct):
    if pct < -100: return '< -100%'
    if pct < -75: return '-100% to -75%'
    if pct < -50: return '-75% to -50%'
    if pct < -25: return '-50% to -25%'
    if pct <= 25: return '-25% to +25%'
    if pct <= 100: return '+25% to +100%'
    return '> +100%'


def summarize_dec(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return ['', '', '', '', '', '', '', '', '']
    return [fmt_dec(min(vals), 4), fmt_dec(q(vals, .05), 4), fmt_dec(q(vals, .25), 4), fmt_dec(q(vals, .5), 4), fmt_dec(q(vals, .75), 4), fmt_dec(q(vals, .95), 4), fmt_dec(max(vals), 4), fmt_dec(sum(vals), 2), fmt_dec(Decimal(str(mean([float(v) for v in vals]))), 4)]


def load():
    with CSV_PATH.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def analyze(fields, rows):
    n = len(rows)
    missing = {f: 0 for f in fields}
    uniques = {f: set() for f in fields}
    top = {f: Counter() for f in fields}
    nums = {f: [] for f in NUMERIC_FIELDS if f in fields}
    bad_nums = {f: 0 for f in nums}
    dates = {f: [] for f in DATE_FIELDS if f in fields}
    bad_dates = {f: 0 for f in dates}
    item = defaultdict(lambda: {
        'rows': 0, 'dates': [], 'unique_dates': set(), 'sum_burn': Decimal(0), 'sum_delta_months': Decimal(0),
        'num_paygroups_values': set(), 'days_values': set(), 'rate_values': set(), 'first_values': set(), 'last_values': set(),
        'neg': 0, 'zero': 0, 'pos': 0, 'pct_values': [], 'burns': [], 'deltas': []
    })
    checks = Counter()
    duplicate_rows = Counter()
    duplicate_item_date = Counter()
    sequence_gaps = []
    row_enriched = []

    for row in rows:
        duplicate_rows[tuple(row.get(f, '') for f in fields)] += 1
        itemid = row['ITEMID']
        posting_dt = parse_dt(row.get('WPPOSTINGDATE'))
        first_dt = parse_dt(row.get('FIRSTPOSTINGDATE'))
        last_dt = parse_dt(row.get('LASTPOSTINGDATE'))
        num_pg = D(row.get('NUM_PAYGROUPS'))
        days_between = D(row.get('DAYS_BETWEEN'))
        rate_months = D(row.get('CI_BURN_RATE_MONTHS'))
        this_burn = D(row.get('THIS_BURN'))
        delta_pct = D(row.get('BURN_DELTA_MONTHS_PERCENT'))
        delta_abs = D(row.get('BURN_DELTA_MONTHS'))

        for f in fields:
            val = row.get(f, '')
            if val is None or str(val).strip() == '':
                missing[f] += 1
            else:
                uniques[f].add(val)
                top[f][val] += 1
        for f in nums:
            val = D(row.get(f))
            if val is None:
                if row.get(f, '').strip(): bad_nums[f] += 1
            else:
                nums[f].append(val)
        for f in dates:
            val = parse_dt(row.get(f))
            if val is None:
                if row.get(f, '').strip(): bad_dates[f] += 1
            else:
                dates[f].append(val)

        x = item[itemid]
        x['rows'] += 1
        if posting_dt:
            x['dates'].append(posting_dt)
            x['unique_dates'].add(posting_dt)
            duplicate_item_date[(itemid, row['WPPOSTINGDATE'])] += 1
        if first_dt: x['first_values'].add(row['FIRSTPOSTINGDATE'])
        if last_dt: x['last_values'].add(row['LASTPOSTINGDATE'])
        if num_pg is not None: x['num_paygroups_values'].add(num_pg)
        if days_between is not None: x['days_values'].add(days_between)
        if rate_months is not None: x['rate_values'].add(rate_months)
        if this_burn is not None:
            x['sum_burn'] += this_burn
            x['burns'].append(this_burn)
            if this_burn < 0: x['neg'] += 1
            elif this_burn == 0: x['zero'] += 1
            else: x['pos'] += 1
        if delta_abs is not None:
            x['sum_delta_months'] += delta_abs
            x['deltas'].append(delta_abs)
        if delta_pct is not None:
            x['pct_values'].append(delta_pct)

        if this_burn is not None and rate_months is not None and delta_abs is not None:
            checks['delta_abs_match' if abs((this_burn - rate_months) - delta_abs) <= Decimal('0.0001') else 'delta_abs_mismatch'] += 1
        if this_burn is not None and rate_months not in (None, Decimal(0)) and delta_pct is not None:
            expected_pct = ((this_burn - rate_months) / rate_months) * Decimal(100)
            checks['delta_pct_match' if abs(expected_pct - delta_pct) <= Decimal('0.0001') else 'delta_pct_mismatch'] += 1
        if posting_dt and first_dt and last_dt:
            checks['posting_in_span' if first_dt <= posting_dt <= last_dt else 'posting_outside_span'] += 1
            checks['first_lte_last' if first_dt <= last_dt else 'first_after_last'] += 1
        if first_dt and last_dt and days_between is not None:
            actual_days = Decimal(str((last_dt - first_dt).total_seconds() / 86400))
            checks['days_between_match' if abs(actual_days - days_between) <= Decimal('1.1') else 'days_between_mismatch'] += 1
        if days_between is not None and rate_months is not None:
            months = (days_between + Decimal(30)) / Decimal(30)
        else:
            months = None
        row_enriched.append({
            'row': row, 'itemid': itemid, 'posting_dt': posting_dt, 'num_pg': int(num_pg) if num_pg is not None else None,
            'days': int(days_between) if days_between is not None else None, 'months': months,
            'rate': rate_months, 'burn': this_burn, 'delta_pct': delta_pct, 'delta_abs': delta_abs,
        })

    for itemid, x in item.items():
        dates_sorted = sorted(x['dates'])
        for prev, curr in zip(dates_sorted, dates_sorted[1:]):
            sequence_gaps.append((curr - prev).total_seconds() / 86400)

    item_rows = []
    for itemid, x in item.items():
        num = next(iter(x['num_paygroups_values'])) if len(x['num_paygroups_values']) == 1 else None
        days = next(iter(x['days_values'])) if len(x['days_values']) == 1 else None
        rate = next(iter(x['rate_values'])) if len(x['rate_values']) == 1 else None
        months = ((days + Decimal(30)) / Decimal(30)) if days is not None else None
        implied_total = rate * months if rate is not None and months is not None else None
        item_rows.append({
            'itemid': itemid, 'rows': x['rows'], 'unique_dates': len(x['unique_dates']), 'num': num, 'days': days, 'months': months,
            'rate': rate, 'sum_burn': x['sum_burn'], 'implied_total': implied_total,
            'implied_error': (x['sum_burn'] - implied_total) if implied_total is not None else None,
            'neg': x['neg'], 'zero': x['zero'], 'pos': x['pos'],
            'min_pct': min(x['pct_values']) if x['pct_values'] else None,
            'max_pct': max(x['pct_values']) if x['pct_values'] else None,
            'median_pct': q(x['pct_values'], .5) if x['pct_values'] else None,
            'abs_pct_mean': Decimal(str(mean([abs(float(v)) for v in x['pct_values']]))) if x['pct_values'] else None,
            'burn_cv': None if not x['burns'] or mean([float(v) for v in x['burns']]) == 0 else Decimal(str((sum((float(v)-mean([float(b) for b in x['burns']]))**2 for v in x['burns'])/len(x['burns']))**0.5 / abs(mean([float(b) for b in x['burns']])))),
        })

    return {
        'n': n, 'missing': missing, 'uniques': uniques, 'top': top, 'nums': nums, 'bad_nums': bad_nums, 'dates': dates,
        'bad_dates': bad_dates, 'item': item, 'item_rows': item_rows, 'checks': checks,
        'duplicate_rows': duplicate_rows, 'duplicate_item_date': duplicate_item_date, 'sequence_gaps': sequence_gaps,
        'row_enriched': row_enriched,
    }


def group_rows(row_enriched, key_func):
    groups = defaultdict(list)
    for r in row_enriched:
        groups[key_func(r)].append(r)
    rows = []
    order = sorted(groups)
    for key in order:
        vals = groups[key]
        pct = [v['delta_pct'] for v in vals if v['delta_pct'] is not None]
        burn = [v['burn'] for v in vals if v['burn'] is not None]
        rows.append([
            key, fmt_int(len(vals)), fmt_dec(q(pct, .25), 2), fmt_dec(q(pct, .5), 2), fmt_dec(q(pct, .75), 2),
            fmt_dec(q(pct, .95), 2), fmt_dec(sum(burn), 2), fmt_pct(sum(1 for v in vals if v['burn'] is not None and v['burn'] < 0) / len(vals))
        ])
    return rows


def build():
    fields, rows = load()
    s = analyze(fields, rows)
    n = s['n']
    item_rows = s['item_rows']
    item_count = len(item_rows)

    field_rows = []
    for f in fields:
        vals = [r.get(f, '') for r in rows if r.get(f, '').strip()]
        examples = '; '.join(list(dict.fromkeys(vals[:5]))[:3])
        role = 'date' if f in DATE_FIELDS else 'monthly model' if f in MONTH_FIELDS else 'pay-group model' if f in PG_FIELDS else 'identifier/count' if f in ['ITEMID', 'NUM_PAYGROUPS', 'DAYS_BETWEEN'] else 'actual burn'
        field_rows.append([f, role, fmt_int(s['missing'][f]), fmt_pct(s['missing'][f]/n), fmt_int(len(s['uniques'][f])), examples, FIELD_NOTES.get(f, '')])

    num_rows = []
    for f in [x for x in NUMERIC_FIELDS if x in fields]:
        num_rows.append([f, fmt_int(len(s['nums'][f])), fmt_int(s['missing'][f]), fmt_int(s['bad_nums'][f])] + summarize_dec(s['nums'][f]))

    date_rows = []
    for f in DATE_FIELDS:
        vals = sorted(s['dates'][f])
        date_rows.append([f, fmt_int(len(vals)), fmt_int(s['missing'][f]), fmt_int(s['bad_dates'][f]), vals[0], vals[len(vals)//2], vals[-1]])

    check_rows = []
    for base in sorted(set(k.rsplit('_', 1)[0] for k in s['checks'])):
        good = s['checks'].get(base + '_match', 0) + s['checks'].get(base + '_span', 0) + s['checks'].get(base + '_last', 0)
        bad = s['checks'].get(base + '_mismatch', 0) + s['checks'].get(base + '_span', 0)
    # explicit checks to avoid ambiguous suffix handling
    check_rows = [
        ['BURN_DELTA_MONTHS = THIS_BURN - CI_BURN_RATE_MONTHS', fmt_int(s['checks']['delta_abs_match']), fmt_int(s['checks']['delta_abs_mismatch'])],
        ['BURN_DELTA_MONTHS_PERCENT formula', fmt_int(s['checks']['delta_pct_match']), fmt_int(s['checks']['delta_pct_mismatch'])],
        ['WPPOSTINGDATE within FIRST/LAST span', fmt_int(s['checks']['posting_in_span']), fmt_int(s['checks']['posting_outside_span'])],
        ['FIRSTPOSTINGDATE <= LASTPOSTINGDATE', fmt_int(s['checks']['first_lte_last']), fmt_int(s['checks']['first_after_last'])],
        ['DAYS_BETWEEN agrees with FIRST/LAST dates', fmt_int(s['checks']['days_between_match']), fmt_int(s['checks']['days_between_mismatch'])],
    ]

    duplicate_exact = sum(c - 1 for c in s['duplicate_rows'].values() if c > 1)
    duplicate_date_groups = [(k, c) for k, c in s['duplicate_item_date'].items() if c > 1]
    row_vs_num_match = sum(1 for x in item_rows if x['num'] is not None and x['rows'] == int(x['num']))
    date_vs_num_match = sum(1 for x in item_rows if x['num'] is not None and x['unique_dates'] == int(x['num']))
    implied_ok = sum(1 for x in item_rows if x['implied_error'] is not None and abs(x['implied_error']) <= Decimal('0.01'))

    item_dist = Counter(int(x['num']) for x in item_rows if x['num'] is not None)
    item_dist_rows = [[k, fmt_int(v), fmt_pct(v/item_count)] for k, v in sorted(item_dist.items())]
    pct_bucket = Counter(bucket_pct(float(r['delta_pct'])) for r in s['row_enriched'] if r['delta_pct'] is not None)
    pct_bucket_order = ['< -100%', '-100% to -75%', '-75% to -50%', '-50% to -25%', '-25% to +25%', '+25% to +100%', '> +100%']
    pct_bucket_rows = [[b, fmt_int(pct_bucket[b]), fmt_pct(pct_bucket[b]/n)] for b in pct_bucket_order]

    npg_group_rows = group_rows(s['row_enriched'], lambda r: bucket_num_paygroups(r['num_pg']))
    day_group_rows = group_rows(s['row_enriched'], lambda r: bucket_days(r['days']))

    top_positive = sorted(s['row_enriched'], key=lambda r: r['delta_pct'] if r['delta_pct'] is not None else Decimal('-999999999'), reverse=True)[:20]
    top_negative = sorted(s['row_enriched'], key=lambda r: r['delta_pct'] if r['delta_pct'] is not None else Decimal('999999999'))[:20]
    outlier_headers = ['ITEMID', 'WPPOSTINGDATE', 'NUM_PAYGROUPS', 'DAYS_BETWEEN', 'CI_BURN_RATE_MONTHS', 'THIS_BURN', 'BURN_DELTA_MONTHS_PERCENT', 'BURN_DELTA_MONTHS']
    def outlier_row(r):
        row = r['row']
        return [row[h] for h in outlier_headers]

    high_items = sorted(item_rows, key=lambda x: abs(float(x['sum_burn'])), reverse=True)[:25]
    high_item_rows = [[x['itemid'], x['rows'], x['unique_dates'], int(x['num']), int(x['days']), fmt_dec(x['months'], 2), fmt_dec(x['rate'], 2), fmt_dec(x['sum_burn'], 2), fmt_dec(x['min_pct'], 2), fmt_dec(x['median_pct'], 2), fmt_dec(x['max_pct'], 2), x['neg']] for x in high_items]
    volatile_items = sorted(item_rows, key=lambda x: x['abs_pct_mean'] if x['abs_pct_mean'] is not None else Decimal('-1'), reverse=True)[:25]
    volatile_rows = [[x['itemid'], x['rows'], x['unique_dates'], int(x['num']), int(x['days']), fmt_dec(x['sum_burn'], 2), fmt_dec(x['abs_pct_mean'], 2), fmt_dec(x['min_pct'], 2), fmt_dec(x['median_pct'], 2), fmt_dec(x['max_pct'], 2), x['neg']] for x in volatile_items]

    neg_rows = [r for r in s['row_enriched'] if r['burn'] is not None and r['burn'] < 0]
    zero_rows = [r for r in s['row_enriched'] if r['burn'] is not None and r['burn'] == 0]
    neg_by_item = Counter(r['itemid'] for r in neg_rows)
    neg_item_rows = [[itemid, fmt_int(count), fmt_pct(count / s['item'][itemid]['rows']), fmt_dec(s['item'][itemid]['sum_burn'], 2)] for itemid, count in neg_by_item.most_common(25)]

    gaps = sorted(s['sequence_gaps'])
    gap_rows = [['Inter-posting gap days', fmt_int(len(gaps)), fmt_dec(gaps[0], 2), fmt_dec(median(gaps), 2), fmt_dec(gaps[-1], 2), fmt_dec(mean(gaps), 2)]] if gaps else []
    dup_item_date_rows = [[k[0], k[1], c] for k, c in sorted(duplicate_date_groups, key=lambda x: x[1], reverse=True)[:30]]

    summary = [
        f'The dataset has {fmt_int(n)} posting rows for {fmt_int(item_count)} contract items.',
        f'Every field is populated in this extract; missingness is not the main issue.',
        f'The monthly delta formulas are internally consistent: {fmt_int(s["checks"]["delta_abs_match"])} absolute-delta rows and {fmt_int(s["checks"]["delta_pct_match"])} percent-delta rows match the stated formulas.',
        f'The monthly linear model is a poor row-level predictor for many postings: median `BURN_DELTA_MONTHS_PERCENT` is {fmt_dec(q(s["nums"]["BURN_DELTA_MONTHS_PERCENT"], .5), 2)}%, and only {fmt_int(sum(1 for r in s["row_enriched"] if abs(r["delta_pct"]) <= 25))} rows are within +/-25% of linear.',
        f'The distribution is asymmetric and bursty: {fmt_int(sum(1 for r in s["row_enriched"] if r["delta_pct"] > 100))} rows exceed +100%, while {fmt_int(sum(1 for r in s["row_enriched"] if r["delta_pct"] < -100))} rows are below -100%.',
        f'Negative actual burn exists in {fmt_int(len(neg_rows))} rows, so reversals/corrections materially affect the lower tail.',
        f'Only {fmt_int(row_vs_num_match)} of {fmt_int(item_count)} items have row count equal to `NUM_PAYGROUPS`; only {fmt_int(date_vs_num_match)} have distinct posting-date count equal to `NUM_PAYGROUPS`. This is the main grain caveat.',
        f'The item-level monthly rate reconciles to item total burn for {fmt_int(implied_ok)} of {fmt_int(item_count)} items using `(DAYS_BETWEEN + 30) / 30`, confirming the broad monthly-rate construction.'
    ]

    cells = [
        md_cell('# CI Payment Details 2 Monthly Burn EDA\n\nThis notebook analyzes `ci_payment_details_2.csv`, focusing on the `_months` model fields as requested. `_pg` fields are retained in the inventory but are not used for primary interpretation.'),
        md_cell('## Brief Analysis and Assumptions\n\n' + '\n\n'.join(summary) + '\n\nClarifying questions carried as assumptions for this pass:\n\n- Should repeated `ITEMID` + `WPPOSTINGDATE` rows be collapsed to one pay group, or retained as separate posting/detail rows? This notebook retains them and separately reports duplicate date groups.\n- Should negative `THIS_BURN` values be treated as valid correction/reversal postings? This notebook treats them as valid and profiles their impact.\n- Is `THIS_BURN` an amount for the posting row rather than a pre-normalized monthly rate? This notebook interprets it as the actual row burn compared to the monthly-linear model.'),
        code_cell("import csv\nfrom collections import Counter, defaultdict\nfrom decimal import Decimal\nfrom datetime import datetime\nfrom pathlib import Path\n\nCSV_PATH = Path('ci_payment_details_2.csv')\nwith CSV_PATH.open(newline='', encoding='utf-8-sig') as handle:\n    reader = csv.DictReader(handle)\n    rows = list(reader)\n    fields = reader.fieldnames\n\nprint(f'Rows: {len(rows):,}')\nprint(f'Fields: {len(fields):,}')\nprint(fields)"),
        md_cell('## Field Inventory\n\n' + table(['Field', 'Role', 'Missing', 'Missing %', 'Unique', 'Examples', 'Interpretation'], field_rows)),
        md_cell('## Numeric Ranges\n\n' + table(['Field', 'Parsed', 'Missing', 'Parse failures', 'Min', 'P05', 'P25', 'Median', 'P75', 'P95', 'Max', 'Sum', 'Mean'], num_rows)),
        md_cell('## Date Ranges\n\n' + table(['Field', 'Parsed', 'Missing', 'Parse failures', 'Min', 'Median', 'Max'], date_rows)),
        md_cell('## Internal Consistency Checks\n\n' + table(['Check', 'Pass', 'Fail'], check_rows) + f'\n\nExact duplicate rows: {fmt_int(duplicate_exact)}. Duplicate `ITEMID` + `WPPOSTINGDATE` groups: {fmt_int(len(duplicate_date_groups))}.'),
        md_cell('## Row Grain and Paygroup Count\n\n`NUM_PAYGROUPS` behaves like an item-level count from the query rather than a guaranteed count of rows in this export. This matters because repeated dates or row-level detail splits can make a posting group appear more than once.\n\n' + table(['NUM_PAYGROUPS', 'Item count', 'Share of items'], item_dist_rows) + f'\n\nItems where row count equals `NUM_PAYGROUPS`: {fmt_int(row_vs_num_match)} of {fmt_int(item_count)}.\n\nItems where distinct posting dates equal `NUM_PAYGROUPS`: {fmt_int(date_vs_num_match)} of {fmt_int(item_count)}.'),
        md_cell('## Monthly Delta Distribution\n\n' + table(['Delta percent bucket', 'Rows', 'Share'], pct_bucket_rows) + '\n\nThe large mass below zero means most posting rows are below the even monthly allocation. The positive tail indicates concentrated/bursty postings that exceed the linear monthly model by multiples.'),
        md_cell('## Delta by Number of Paygroups\n\n' + table(['NUM_PAYGROUPS bucket', 'Rows', 'P25 delta %', 'Median delta %', 'P75 delta %', 'P95 delta %', 'Total this burn', 'Negative burn share'], npg_group_rows)),
        md_cell('## Delta by Item Duration\n\n' + table(['DAYS_BETWEEN bucket', 'Rows', 'P25 delta %', 'Median delta %', 'P75 delta %', 'P95 delta %', 'Total this burn', 'Negative burn share'], day_group_rows)),
        md_cell('## Top Positive Monthly Delta Rows\n\nThese are the posting rows most above the monthly-linear model.\n\n' + table(outlier_headers, [outlier_row(r) for r in top_positive])),
        md_cell('## Top Negative Monthly Delta Rows\n\nThese rows are most below the monthly-linear model. Values below -100% are possible because actual burn can be negative.\n\n' + table(outlier_headers, [outlier_row(r) for r in top_negative])),
        md_cell('## Largest Items by Absolute Total Burn\n\n' + table(['ITEMID', 'Rows', 'Unique posting dates', 'NUM_PAYGROUPS', 'DAYS_BETWEEN', 'Modeled months', 'CI burn rate months', 'Sum this burn', 'Min delta %', 'Median delta %', 'Max delta %', 'Negative rows'], high_item_rows)),
        md_cell('## Most Volatile Items by Mean Absolute Delta Percent\n\n' + table(['ITEMID', 'Rows', 'Unique posting dates', 'NUM_PAYGROUPS', 'DAYS_BETWEEN', 'Sum this burn', 'Mean abs delta %', 'Min delta %', 'Median delta %', 'Max delta %', 'Negative rows'], volatile_rows)),
        md_cell('## Negative and Zero Burn Rows\n\n' + table([ 'Metric', 'Value'], [['Negative THIS_BURN rows', fmt_int(len(neg_rows))], ['Zero THIS_BURN rows', fmt_int(len(zero_rows))], ['Items with at least one negative row', fmt_int(len(neg_by_item))]]) + '\n\n### Items With Most Negative Rows\n\n' + table(['ITEMID', 'Negative rows', 'Negative row share', 'Item total this burn'], neg_item_rows)),
        md_cell('## Posting-Date Spacing\n\n' + table(['Metric', 'Count', 'Min', 'Median', 'Max', 'Mean'], gap_rows) + '\n\n### Duplicate ITEMID + WPPOSTINGDATE Groups\n\n' + table(['ITEMID', 'WPPOSTINGDATE', 'Rows on same date'], dup_item_date_rows)),
        md_cell('## Modeling Implications\n\n- The monthly-linear model is useful as a baseline, but it should not be expected to predict individual posting rows closely. The empirical pattern is under-burn in many periods and sharp over-burn in fewer concentrated periods.\n- `BURN_DELTA_MONTHS_PERCENT` is a better cross-item comparability field than the absolute delta, but it still explodes for low baseline rates and for negative/correction rows. Winsorized or bucketed views will likely be more stable than raw percent deltas for model training.\n- Treat `ITEMID` as the primary grouping key. Any train/test split or validation should avoid splitting rows from the same item across train and test if the goal is generalization to unseen items.\n- Before modeling payment timing, decide whether repeated `ITEMID` + `WPPOSTINGDATE` rows are separate observations or should be collapsed to a date/paygroup level. That choice changes the interpretation of `NUM_PAYGROUPS` and row weights.\n- Negative burns should be modeled or flagged explicitly; excluding them would remove genuine correction behavior and would also hide why the percent delta can fall below -100%.'),
    ]

    nb = {
        'cells': cells,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}},
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    OUT_PATH.write_text(json.dumps(nb, indent=2, default=str), encoding='utf-8')
    print(f'Wrote {OUT_PATH} with {len(cells)} cells')

if __name__ == '__main__':
    build()
