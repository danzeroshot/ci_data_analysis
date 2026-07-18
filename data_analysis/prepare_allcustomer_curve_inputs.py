#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path('custpaydetails_clustered_cumulative_curves_allcustomers_2026-06-05-0923.csv')
ALL_OUT = Path('clustered_data_input_allcustomers.csv')
PROFILE_OUT = Path('clustered_data_input_allcustomers_profile.csv')


def safe_name(value: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in value).strip('_')


def as_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return float('nan')


def main():
    with SOURCE.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    if 'CUSTOMERNAME' not in fields:
        raise ValueError('Expected CUSTOMERNAME in all-customer clustered curve file')

    with ALL_OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_customer = defaultdict(list)
    for row in rows:
        by_customer[row['CUSTOMERNAME']].append(row)

    profile_rows = []
    for customer in sorted(by_customer):
        cust_rows = by_customer[customer]
        out = Path(f'clustered_data_input_{safe_name(customer)}.csv')
        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(cust_rows)

        by_item = defaultdict(list)
        for row in cust_rows:
            by_item[row['ITEMID']].append(row)
        split_rows = Counter(row['TRAINSPLIT'].lower() for row in cust_rows)
        train_items = {row['ITEMID'] for row in cust_rows if row['TRAINSPLIT'].lower() != 'test'}
        test_items = {row['ITEMID'] for row in cust_rows if row['TRAINSPLIT'].lower() == 'test'}
        y = [as_float(row, 'CUMULATIVEBURNPCT') for row in cust_rows]
        x = [as_float(row, 'ELAPSEDPCT') for row in cust_rows]
        negative_clusters = sum(as_float(row, 'CLUSTERBURN') < 0 for row in cust_rows)
        decreasing_items = 0
        for item_rows in by_item.values():
            prev = None
            for row in sorted(item_rows, key=lambda r: int(float(r['CLUSTERSEQUENCE']))):
                cur = as_float(row, 'CUMULATIVEBURNPCT')
                if prev is not None and cur + 1e-9 < prev:
                    decreasing_items += 1
                    break
                prev = cur
        profile_rows.append({
            'CustomerName': customer,
            'PreparedCsv': out.name,
            'Rows': len(cust_rows),
            'Items': len(by_item),
            'TrainRows': split_rows.get('train', 0),
            'TestRows': split_rows.get('test', 0),
            'TrainItems': len(train_items),
            'TestItems': len(test_items),
            'MinElapsedPct': min(x),
            'MaxElapsedPct': max(x),
            'MinCumulativeBurnPct': min(y),
            'MaxCumulativeBurnPct': max(y),
            'NegativeClusterRows': negative_clusters,
            'ItemsWithDecreasingCumulativePct': decreasing_items,
        })

    all_items = {(row['CUSTOMERNAME'], row['ITEMID']) for row in rows}
    split_rows = Counter(row['TRAINSPLIT'].lower() for row in rows)
    profile_rows.insert(0, {
        'CustomerName': 'ALL_CUSTOMERS',
        'PreparedCsv': ALL_OUT.name,
        'Rows': len(rows),
        'Items': len(all_items),
        'TrainRows': split_rows.get('train', 0),
        'TestRows': split_rows.get('test', 0),
        'TrainItems': len({(row['CUSTOMERNAME'], row['ITEMID']) for row in rows if row['TRAINSPLIT'].lower() != 'test'}),
        'TestItems': len({(row['CUSTOMERNAME'], row['ITEMID']) for row in rows if row['TRAINSPLIT'].lower() == 'test'}),
        'MinElapsedPct': min(as_float(row, 'ELAPSEDPCT') for row in rows),
        'MaxElapsedPct': max(as_float(row, 'ELAPSEDPCT') for row in rows),
        'MinCumulativeBurnPct': min(as_float(row, 'CUMULATIVEBURNPCT') for row in rows),
        'MaxCumulativeBurnPct': max(as_float(row, 'CUMULATIVEBURNPCT') for row in rows),
        'NegativeClusterRows': sum(as_float(row, 'CLUSTERBURN') < 0 for row in rows),
        'ItemsWithDecreasingCumulativePct': '',
    })

    with PROFILE_OUT.open('w', newline='', encoding='utf-8') as f:
        fields_out = list(profile_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields_out)
        writer.writeheader()
        writer.writerows(profile_rows)

    print(f'wrote {ALL_OUT}')
    for row in profile_rows[1:]:
        print(f"wrote {row['PreparedCsv']} ({row['Rows']} rows, {row['Items']} items)")
    print(f'wrote {PROFILE_OUT}')


if __name__ == '__main__':
    main()
