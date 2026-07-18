#!/usr/bin/env python3
"""Flatten Snowflake OBJECT_CONSTRUCT(*) CSV exports.

Input shape:
    RECORD_ID, ALL_COLUMNS_JSON

Output shape:
    one ordinary flat CSV column per JSON object key, preserving RECORD_ID first.

The script does two streaming passes:
    1. discover the union of JSON keys and basic row/schema stats
    2. write the flattened CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float)):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--json-column", default="ALL_COLUMNS_JSON")
    parser.add_argument("--record-id-column", default="RECORD_ID")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raise_csv_limit()

    key_order: list[str] = []
    seen: set[str] = set()
    row_count = 0
    max_json_len = 0

    with args.input_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header")
        missing = [c for c in [args.record_id_column, args.json_column] if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing expected columns: {missing}; found {reader.fieldnames}")

        for row in reader:
            row_count += 1
            raw = row[args.json_column]
            max_json_len = max(max_json_len, len(raw))
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSON parse failed on input row {row_count}: {exc}") from exc
            if not isinstance(obj, dict):
                raise SystemExit(f"JSON payload on row {row_count} is {type(obj).__name__}, expected object")
            for key in obj.keys():
                if key not in seen:
                    seen.add(key)
                    key_order.append(key)
            if args.progress_every and row_count % args.progress_every == 0:
                print(f"schema pass rows={row_count:,} columns={len(key_order):,}", file=sys.stderr)

    # Snowflake object keys are already column names. Keep discovered order, but avoid duplicating RECORD_ID.
    json_keys = [k for k in key_order if k != args.record_id_column]
    output_columns = [args.record_id_column] + json_keys

    written = 0
    with args.input_csv.open("r", newline="", encoding="utf-8-sig") as src, args.output_csv.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            obj = json.loads(row[args.json_column])
            out = {key: normalize_value(obj.get(key)) for key in json_keys}
            out[args.record_id_column] = row[args.record_id_column]
            writer.writerow(out)
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                print(f"write pass rows={written:,}", file=sys.stderr)

    print(f"input_rows={row_count}")
    print(f"output_rows={written}")
    print(f"output_columns={len(output_columns)}")
    print(f"max_json_field_chars={max_json_len}")
    print(f"output_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
