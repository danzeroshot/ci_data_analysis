#!/usr/bin/env python3
"""Generate a self-contained EDA notebook for custpaydetails.csv.csv.

The workspace does not include pandas/matplotlib, so this intentionally uses
only the standard library plus the already-installed nbformat-compatible JSON
format.
"""

from __future__ import annotations

import csv
import json
import math
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median


CSV_PATH = Path("custpaydetails.csv.csv")
SQL_PATH = Path("custpaydetails.sql")
OUT_PATH = Path("custpaydetails_eda.ipynb")

DATE_FIELDS = [
    "CONTRACTSTARTDATE",
    "CONTRACTENDDATE",
    "PROJSTARTDATE",
    "PROJENDDATE",
    "PHASEITEMSTARTDATE",
    "PHASEITEMENDDATE",
    "WPPOSTINGDATE",
    "POSTINGDATE",
    "FROMDATE",
    "TODATE",
]

NUMERIC_FIELDS = [
    "CONTRACTID",
    "ITEMID",
    "ITEMUNITPRICE",
    "ITEMTOTALQUANTITY",
    "LINKEDBUDGETITEMID",
    "POSTINGID",
    "POSTINGQTY",
    "POSTINGUNITRATE",
    "TOTALPOSTINGAMOUNT",
    "REFERENCEPOSTINGID",
    "PAYAPPID",
    "PAYAPPSEQUENCE",
    "PEDETAILSID",
    "BILLINGRATE",
    "QTYBILLEDTHISPAYAPP",
    "BILLEDAMOUNT",
    "TOTALWORKCOMPLETED",
]

ID_FIELDS = {
    "CONTRACTID",
    "ITEMID",
    "LINKEDBUDGETITEMID",
    "POSTINGID",
    "REFERENCEPOSTINGID",
    "PAYAPPID",
    "PEDETAILSID",
}

MONEY_FIELDS = {
    "ITEMUNITPRICE",
    "POSTINGUNITRATE",
    "TOTALPOSTINGAMOUNT",
    "BILLINGRATE",
    "BILLEDAMOUNT",
    "TOTALWORKCOMPLETED",
}

SOURCE_CONTEXT = {
    "CUSTOMER": "Literal from query. Active export is the UDOT CTE only.",
    "PROJECTNAME": "PROJECTPROJECTMAIN.PROJECTNAME.",
    "PROJECTCODE": "PROJECTPROJECTMAIN.PROJECTCODE.",
    "PROJECTDESC": "PROJECTPROJECTMAIN.DESCRIPTION.",
    "PROJECTTYPE": "LibraryProjectType.PROJECTTYPE via PM.ProjectTypeID.",
    "PROJECTSTATUS": "PROJECTProjectStatus.StatusName via PM.StatusId.",
    "CONTRACTID": "CONTMGTMaster.ID joined from CORITEMItemDetails.PARENTINSTANCEID.",
    "CONTRACTNAME": "CONTMGTMaster.NAME.",
    "CONTRACTDESCRIPTION": "CONTMGTMaster.DESC.",
    "CONTRACTSTARTDATE": "CONTMGTMaster.StartDt.",
    "CONTRACTENDDATE": "CONTMGTMaster.CLOSUREDT; query filters ContractEndDate < CURRENT_DATE.",
    "PROJSTARTDATE": "PROJECTPROJECTMAIN.STARTDATE.",
    "PROJENDDATE": "PROJECTPROJECTMAIN.ENDDATE.",
    "PHASEITEMSTARTDATE": "PROJECTProjectPhaseItem.STARTDATE through contract-item container hierarchy.",
    "PHASEITEMENDDATE": "PROJECTProjectPhaseItem.ENDDATE through contract-item container hierarchy.",
    "WPPOSTINGDATE": "PROCMGTWorkPosting.POSTINGDATE, selected separately from POSTINGDATE.",
    "CONTRACTTYPE": "LibraryTypeOfContract.TypeOfContract.",
    "ITEMID": "CORITEMItemDetails.ItemID for the contract pay item.",
    "STANDARDITEMNO": "CORITEMItemDetails.StandardItemNo for the contract pay item.",
    "ITEMNAME": "CORITEMItemDetails.DESCRIPTION for the contract pay item.",
    "ITEMUNITPRICE": "CORITEMItemDetails.UnitPrice.",
    "ITEMTOTALQUANTITY": "CORITEMItemDetails.ContractQuantity.",
    "CONTRACTITEMCONTAINERPATH": "Recursive CORITEMContainer full path for CONTMGT.",
    "CONTRACTITEMCONTAINERNAME": "Leaf CORITEMContainer.ContainerName for CONTMGT.",
    "CONTRACTITEMSYSTEMNAME": "projectmeasurementsystems.SYSTEMNAME through contract measurement system.",
    "CONTRACTITEMUNIT": "PROJECTMEASUREMENTUNITS.Unit.",
    "CONTRACTITEMPHASENAME": "PROJECTProjectPhaseItem.Phase through contract-item container hierarchy.",
    "LINKEDBUDGETITEMID": "Linked CORITEMItemDetails.ItemID when CI.LinkedBudgetItem points to BDGTEST/BDGTREV.",
    "LINKEDBUDGETITEM": "Linked budget item StandardItemNo.",
    "LINKEDBUDGETITEMMODULE": "Linked budget item ModuleID, expected BDGTEST or BDGTREV when present.",
    "LINKEDBUDGETITEMDESCRIPTION": "Linked budget item DESCRIPTION.",
    "LINKEDBUDGETITEMCONTAINERPATH": "Recursive CORITEMContainer full path for linked budget item.",
    "LINKEDBUDGETITEMCONTAINERNAME": "Leaf budget container name.",
    "LINKEDBUDGETITEMPHASENAME": "PROJECTProjectPhaseItem.Phase for linked budget item.",
    "POSTINGID": "PROCMGTWorkPosting.WPostingID.",
    "POSTINGQTY": "PROCMGTWorkPosting.PostingQty.",
    "POSTINGUNITRATE": "PROCMGTWorkPosting.UnitRate.",
    "TOTALPOSTINGAMOUNT": "Computed in SQL as WP.PostingQty * WP.UnitRate.",
    "REFERENCEPOSTINGTYPE": "PROCMGTWorkPosting.ReferencePostingType; join restricts this to ITMPOST.",
    "REFERENCEPOSTINGID": "PROCMGTWorkPosting.ReferencePostingID.",
    "POSTINGSTATUS": "PROCMGTWorkPosting.Status; query filters to non-null.",
    "POSTINGDATE": "PROCMGTWorkPosting.PostingDate.",
    "PAYAPPID": "PROCMGTPayEstimates.PEID through PEDetails.",
    "PAYAPPNUMBER": "PROCMGTPayEstimates.PayEstimateNumber.",
    "PAYAPPSEQUENCE": "PROCMGTPayEstimates.PENum.",
    "PAYAPPSTATUS": "PROCMGTPayEstimates.Status; query filters to non-null.",
    "FROMDATE": "PROCMGTPayEstimates.FromDate.",
    "TODATE": "PROCMGTPayEstimates.ToDate.",
    "PEDETAILSID": "PROCMGTPEDetails.PEDetailsID.",
    "BILLINGRATE": "PROCMGTPEDetails.BillingRate.",
    "QTYBILLEDTHISPAYAPP": "PROCMGTPEDetails.Quantity.",
    "BILLEDAMOUNT": "PROCMGTPEDetails.Amount.",
    "TOTALWORKCOMPLETED": "Computed in SQL as CI.UnitPrice * PED.Quantity.",
}


def is_missing(value: str | None) -> bool:
    return value is None or value.strip() == ""


def parse_decimal(value: str | None) -> Decimal | None:
    if is_missing(value):
        return None
    try:
        return Decimal(value.strip())
    except InvalidOperation:
        return None


def parse_datetime(value: str | None) -> datetime | None:
    if is_missing(value):
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def fmt_dec(value: Decimal | float | int | None, places: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return f"{float(value):,.{places}f}"
    return f"{value:,.{places}f}"


def escape_md(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "<br>").replace("|", "\\|")
    return text


def md_table(headers: list[str], rows: list[list[object]], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in shown:
        out.append("| " + " | ".join(escape_md(v) for v in row) + " |")
    if max_rows is not None and len(rows) > max_rows:
        out.append("| " + " | ".join([f"... {len(rows) - max_rows:,} more rows"] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(out)


def quantiles(values: list[Decimal], probs: list[float]) -> dict[float, Decimal]:
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    result = {}
    for p in probs:
        if n == 1:
            result[p] = ordered[0]
            continue
        pos = p * (n - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            result[p] = ordered[lo]
        else:
            result[p] = ordered[lo] + (ordered[hi] - ordered[lo]) * Decimal(str(pos - lo))
    return result


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source}


def load_rows() -> tuple[list[str], list[dict[str, str]]]:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def profile_data(fields: list[str], rows: list[dict[str, str]]) -> dict:
    n = len(rows)
    missing = {field: 0 for field in fields}
    unique = {field: set() for field in fields}
    top_values = {field: Counter() for field in fields}
    samples = {field: [] for field in fields}
    numeric_values = {field: [] for field in NUMERIC_FIELDS if field in fields}
    numeric_bad = {field: 0 for field in NUMERIC_FIELDS if field in fields}
    date_values = {field: [] for field in DATE_FIELDS if field in fields}
    date_bad = {field: 0 for field in DATE_FIELDS if field in fields}
    group_counts = defaultdict(Counter)
    missing_by_group = {field: defaultdict(Counter) for field in fields}
    co_missing = {field: Counter() for field in fields}
    duplicate_rows = Counter()
    item_payapps = defaultdict(set)
    item_rows = Counter()
    contract_items = defaultdict(set)
    payapp_rows = Counter()
    payapp_periods = {}
    standard_prefix = Counter()
    standard_prefix_by_amount = defaultdict(Decimal)
    item_total_billed = defaultdict(Decimal)
    amount_checks = Counter()
    date_checks = Counter()
    date_diffs = defaultdict(list)

    context_fields = [
        "CUSTOMER",
        "PROJECTSTATUS",
        "CONTRACTTYPE",
        "CONTRACTITEMUNIT",
        "LINKEDBUDGETITEMMODULE",
        "PAYAPPSTATUS",
        "POSTINGSTATUS",
        "PROJECTTYPE",
    ]

    for row in rows:
        duplicate_rows[tuple(row.get(field, "") for field in fields)] += 1
        for context in context_fields:
            if context in fields:
                group_value = row.get(context, "")
                if is_missing(group_value):
                    group_value = "<missing>"
                group_counts[context][group_value] += 1

        row_missing = []
        for field in fields:
            value = row.get(field, "")
            if is_missing(value):
                missing[field] += 1
                row_missing.append(field)
            else:
                unique[field].add(value)
                top_values[field][value] += 1
                if len(samples[field]) < 5 and value not in samples[field]:
                    samples[field].append(value)

        for field in row_missing:
            for context in context_fields:
                if context in fields:
                    group_value = row.get(context, "")
                    if is_missing(group_value):
                        group_value = "<missing>"
                    missing_by_group[field][context][group_value] += 1
            for other in row_missing:
                if other != field:
                    co_missing[field][other] += 1

        parsed_numbers = {}
        for field in numeric_values:
            value = parse_decimal(row.get(field))
            if value is None:
                if not is_missing(row.get(field)):
                    numeric_bad[field] += 1
            else:
                numeric_values[field].append(value)
                parsed_numbers[field] = value

        parsed_dates = {}
        for field in date_values:
            value = parse_datetime(row.get(field))
            if value is None:
                if not is_missing(row.get(field)):
                    date_bad[field] += 1
            else:
                date_values[field].append(value)
                parsed_dates[field] = value

        item = row.get("ITEMID")
        payapp = row.get("PAYAPPID")
        contract = row.get("CONTRACTID")
        if not is_missing(item):
            item_rows[item] += 1
            if not is_missing(payapp):
                item_payapps[item].add(payapp)
            if not is_missing(contract):
                contract_items[contract].add(item)
            amt = parsed_numbers.get("BILLEDAMOUNT")
            if amt is not None:
                item_total_billed[item] += amt
        if not is_missing(payapp):
            payapp_rows[payapp] += 1
            if "FROMDATE" in parsed_dates or "TODATE" in parsed_dates:
                payapp_periods.setdefault(payapp, (row.get("FROMDATE", ""), row.get("TODATE", "")))

        standard = row.get("STANDARDITEMNO", "")
        if not is_missing(standard):
            prefix = standard[:5]
            standard_prefix[prefix] += 1
            amt = parsed_numbers.get("BILLEDAMOUNT")
            if amt is not None:
                standard_prefix_by_amount[prefix] += amt

        pq = parsed_numbers.get("POSTINGQTY")
        pur = parsed_numbers.get("POSTINGUNITRATE")
        tpa = parsed_numbers.get("TOTALPOSTINGAMOUNT")
        br = parsed_numbers.get("BILLINGRATE")
        qbp = parsed_numbers.get("QTYBILLEDTHISPAYAPP")
        billed = parsed_numbers.get("BILLEDAMOUNT")
        unit_price = parsed_numbers.get("ITEMUNITPRICE")
        twc = parsed_numbers.get("TOTALWORKCOMPLETED")
        if pq is not None and pur is not None and tpa is not None:
            if abs((pq * pur) - tpa) <= Decimal("0.01"):
                amount_checks["posting_amount_matches"] += 1
            else:
                amount_checks["posting_amount_mismatch"] += 1
        if br is not None and qbp is not None and billed is not None:
            if abs((br * qbp) - billed) <= Decimal("0.01"):
                amount_checks["billed_amount_matches"] += 1
            else:
                amount_checks["billed_amount_mismatch"] += 1
        if unit_price is not None and qbp is not None and twc is not None:
            if abs((unit_price * qbp) - twc) <= Decimal("0.01"):
                amount_checks["total_work_completed_matches"] += 1
            else:
                amount_checks["total_work_completed_mismatch"] += 1

        def add_date_check(name: str, ok: bool):
            date_checks[f"{name}_{'ok' if ok else 'bad'}"] += 1

        if "WPPOSTINGDATE" in parsed_dates and "POSTINGDATE" in parsed_dates:
            delta = (parsed_dates["WPPOSTINGDATE"] - parsed_dates["POSTINGDATE"]).total_seconds() / 86400
            date_diffs["WPPOSTINGDATE_minus_POSTINGDATE_days"].append(delta)
            add_date_check("wpposting_equals_posting", abs(delta) < 0.00001)
        if "FROMDATE" in parsed_dates and "TODATE" in parsed_dates:
            add_date_check("payapp_from_lte_to", parsed_dates["FROMDATE"] <= parsed_dates["TODATE"])
            date_diffs["TODATE_minus_FROMDATE_days"].append((parsed_dates["TODATE"] - parsed_dates["FROMDATE"]).total_seconds() / 86400)
        if "CONTRACTSTARTDATE" in parsed_dates and "CONTRACTENDDATE" in parsed_dates:
            add_date_check("contract_start_lte_end", parsed_dates["CONTRACTSTARTDATE"] <= parsed_dates["CONTRACTENDDATE"])
            date_diffs["CONTRACTENDDATE_minus_CONTRACTSTARTDATE_days"].append((parsed_dates["CONTRACTENDDATE"] - parsed_dates["CONTRACTSTARTDATE"]).total_seconds() / 86400)
        if "PROJSTARTDATE" in parsed_dates and "PROJENDDATE" in parsed_dates:
            add_date_check("project_start_lte_end", parsed_dates["PROJSTARTDATE"] <= parsed_dates["PROJENDDATE"])
            date_diffs["PROJENDDATE_minus_PROJSTARTDATE_days"].append((parsed_dates["PROJENDDATE"] - parsed_dates["PROJSTARTDATE"]).total_seconds() / 86400)
        if "PHASEITEMSTARTDATE" in parsed_dates and "PHASEITEMENDDATE" in parsed_dates:
            add_date_check("phase_start_lte_end", parsed_dates["PHASEITEMSTARTDATE"] <= parsed_dates["PHASEITEMENDDATE"])
            date_diffs["PHASEITEMENDDATE_minus_PHASEITEMSTARTDATE_days"].append((parsed_dates["PHASEITEMENDDATE"] - parsed_dates["PHASEITEMSTARTDATE"]).total_seconds() / 86400)
        if "POSTINGDATE" in parsed_dates and "FROMDATE" in parsed_dates and "TODATE" in parsed_dates:
            add_date_check("posting_inside_payapp_window", parsed_dates["FROMDATE"] <= parsed_dates["POSTINGDATE"] <= parsed_dates["TODATE"])
        if "CONTRACTSTARTDATE" in parsed_dates and "CONTRACTENDDATE" in parsed_dates and "POSTINGDATE" in parsed_dates:
            add_date_check("posting_inside_contract_window", parsed_dates["CONTRACTSTARTDATE"] <= parsed_dates["POSTINGDATE"] <= parsed_dates["CONTRACTENDDATE"])
        if "PROJSTARTDATE" in parsed_dates and "PROJENDDATE" in parsed_dates and "POSTINGDATE" in parsed_dates:
            add_date_check("posting_inside_project_window", parsed_dates["PROJSTARTDATE"] <= parsed_dates["POSTINGDATE"] <= parsed_dates["PROJENDDATE"])

    duplicate_count = sum(count - 1 for count in duplicate_rows.values() if count > 1)
    duplicate_groups = sum(1 for count in duplicate_rows.values() if count > 1)

    return {
        "n": n,
        "missing": missing,
        "unique": unique,
        "top_values": top_values,
        "samples": samples,
        "numeric_values": numeric_values,
        "numeric_bad": numeric_bad,
        "date_values": date_values,
        "date_bad": date_bad,
        "group_counts": group_counts,
        "missing_by_group": missing_by_group,
        "co_missing": co_missing,
        "duplicate_count": duplicate_count,
        "duplicate_groups": duplicate_groups,
        "item_payapps": item_payapps,
        "item_rows": item_rows,
        "contract_items": contract_items,
        "payapp_rows": payapp_rows,
        "payapp_periods": payapp_periods,
        "standard_prefix": standard_prefix,
        "standard_prefix_by_amount": standard_prefix_by_amount,
        "item_total_billed": item_total_billed,
        "amount_checks": amount_checks,
        "date_checks": date_checks,
        "date_diffs": date_diffs,
    }


def numeric_summary(values: list[Decimal]) -> list[str]:
    if not values:
        return ["", "", "", "", "", "", "", ""]
    qs = quantiles(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return [
        fmt_dec(min(values), 4),
        fmt_dec(qs[0.05], 4),
        fmt_dec(qs[0.25], 4),
        fmt_dec(qs[0.5], 4),
        fmt_dec(qs[0.75], 4),
        fmt_dec(qs[0.95], 4),
        fmt_dec(max(values), 4),
        fmt_dec(sum(values), 2),
    ]


def date_summary(values: list[datetime]) -> list[str]:
    if not values:
        return ["", "", "", "", ""]
    ordered = sorted(values)
    return [
        ordered[0].isoformat(sep=" "),
        ordered[len(ordered) // 4].isoformat(sep=" "),
        ordered[len(ordered) // 2].isoformat(sep=" "),
        ordered[(len(ordered) * 3) // 4].isoformat(sep=" "),
        ordered[-1].isoformat(sep=" "),
    ]


def infer_type(field: str) -> str:
    if field in DATE_FIELDS:
        return "date/time"
    if field in MONEY_FIELDS:
        return "money/rate"
    if field in ID_FIELDS:
        return "identifier"
    if field in NUMERIC_FIELDS:
        return "quantity/number"
    return "categorical/text"


def missing_context_text(field: str, stats: dict, total_rows: int) -> str:
    miss = stats["missing"][field]
    if miss == 0:
        return "Never missing in this extract."
    if miss == total_rows:
        return "Always missing in this extract."

    candidates = []
    for context, counts in stats["group_counts"].items():
        for group_value, group_n in counts.items():
            if group_n < 50:
                continue
            group_missing = stats["missing_by_group"][field][context][group_value]
            rate = group_missing / group_n
            overall = miss / total_rows
            if group_missing and abs(rate - overall) >= 0.15:
                candidates.append((rate, group_missing, group_n, context, group_value))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    parts = [f"Missing in {fmt_int(miss)} rows ({fmt_pct(miss / total_rows)})."]
    if candidates:
        rate, group_missing, group_n, context, group_value = candidates[0]
        parts.append(
            f"Highest clear concentration: {context}={group_value} has {fmt_int(group_missing)}/{fmt_int(group_n)} missing ({fmt_pct(rate)})."
        )
    related = stats["co_missing"][field].most_common(3)
    if related:
        parts.append("Often co-missing with " + ", ".join(f"{name} ({fmt_int(count)})" for name, count in related) + ".")
    return " ".join(parts)


def build_findings(fields: list[str], stats: dict) -> dict[str, str]:
    n = stats["n"]
    lines = []
    lines.append(f"The extract contains {fmt_int(n)} rows and {len(fields)} fields.")
    lines.append(
        f"It is a UDOT-only result set: CUSTOMER has {len(stats['unique']['CUSTOMER'])} unique value(s), matching the active CTE in the SQL file."
    )
    lines.append(
        "The SQL requires non-null work posting and pay estimate statuses, so posting/pay-app fields are expected to be dense while left-joined descriptive fields can be sparse."
    )
    dup_count = stats["duplicate_count"]
    lines.append(f"Exact duplicate rows: {fmt_int(dup_count)} across {fmt_int(stats['duplicate_groups'])} duplicate groups.")

    always_missing = [f for f in fields if stats["missing"][f] == n]
    dense = [f for f in fields if stats["missing"][f] == 0]
    lines.append(f"Always-missing fields: {', '.join(always_missing) if always_missing else 'none'}.")
    lines.append(f"Fields with no missing values: {len(dense)} of {len(fields)}.")

    return {"overview": "\n\n".join(lines)}


def build_notebook(fields: list[str], rows: list[dict[str, str]], stats: dict) -> dict:
    n = stats["n"]
    sql_text = SQL_PATH.read_text(encoding="utf-8")
    active_tables = sorted(set(re.findall(r"UDOTAIML\.DBO\.([A-Za-z0-9_]+)", sql_text)))
    findings = build_findings(fields, stats)

    field_rows = []
    for field in fields:
        miss = stats["missing"][field]
        examples = "; ".join(stats["samples"][field][:3])
        top = ", ".join(f"{value} ({fmt_int(count)})" for value, count in stats["top_values"][field].most_common(3))
        field_rows.append(
            [
                field,
                infer_type(field),
                fmt_int(miss),
                fmt_pct(miss / n),
                fmt_int(len(stats["unique"][field])),
                examples,
                top,
                SOURCE_CONTEXT.get(field, ""),
                missing_context_text(field, stats, n),
            ]
        )

    missing_rows = sorted(
        [[field, fmt_int(stats["missing"][field]), fmt_pct(stats["missing"][field] / n), fmt_int(len(stats["unique"][field]))] for field in fields],
        key=lambda row: float(row[2].rstrip("%")),
        reverse=True,
    )

    date_rows = []
    for field in DATE_FIELDS:
        if field not in fields:
            continue
        values = stats["date_values"][field]
        row = [field, fmt_int(len(values)), fmt_int(stats["missing"][field]), fmt_int(stats["date_bad"][field])]
        row += date_summary(values)
        row.append(SOURCE_CONTEXT.get(field, ""))
        date_rows.append(row)

    numeric_rows = []
    for field in NUMERIC_FIELDS:
        if field not in fields:
            continue
        values = stats["numeric_values"][field]
        row = [field, fmt_int(len(values)), fmt_int(stats["missing"][field]), fmt_int(stats["numeric_bad"][field])]
        row += numeric_summary(values)
        numeric_rows.append(row)

    categorical_rows = []
    for field in fields:
        if field in NUMERIC_FIELDS or field in DATE_FIELDS:
            continue
        top = stats["top_values"][field].most_common(8)
        categorical_rows.append(
            [
                field,
                fmt_int(stats["missing"][field]),
                fmt_pct(stats["missing"][field] / n),
                fmt_int(len(stats["unique"][field])),
                "; ".join(f"{value} ({fmt_int(count)}, {fmt_pct(count / n)})" for value, count in top),
            ]
        )

    date_check_rows = []
    check_names = sorted({key.rsplit("_", 1)[0] for key in stats["date_checks"]})
    for name in check_names:
        ok = stats["date_checks"].get(f"{name}_ok", 0)
        bad = stats["date_checks"].get(f"{name}_bad", 0)
        total = ok + bad
        if total:
            date_check_rows.append([name, fmt_int(ok), fmt_int(bad), fmt_pct(bad / total)])

    duration_rows = []
    for name, values in sorted(stats["date_diffs"].items()):
        if values:
            ordered = sorted(values)
            duration_rows.append(
                [
                    name,
                    fmt_int(len(values)),
                    f"{ordered[0]:,.2f}",
                    f"{median(ordered):,.2f}",
                    f"{ordered[-1]:,.2f}",
                    f"{mean(ordered):,.2f}",
                ]
            )

    amount_check_rows = []
    for base in ["posting_amount", "billed_amount", "total_work_completed"]:
        matches = stats["amount_checks"].get(f"{base}_matches", 0)
        mismatches = stats["amount_checks"].get(f"{base}_mismatch", 0)
        total = matches + mismatches
        if total:
            amount_check_rows.append([base, fmt_int(matches), fmt_int(mismatches), fmt_pct(mismatches / total)])

    top_contract_rows = [
        [contract, fmt_int(len(items))]
        for contract, items in sorted(stats["contract_items"].items(), key=lambda item: len(item[1]), reverse=True)[:20]
    ]
    payapp_distribution = Counter(len(payapps) for payapps in stats["item_payapps"].values())
    payapp_dist_rows = [
        [num_payapps, fmt_int(count), fmt_pct(count / max(1, len(stats["item_payapps"])))]
        for num_payapps, count in sorted(payapp_distribution.items())[:30]
    ]
    top_item_amount_rows = [
        [item, fmt_dec(amount, 2), fmt_int(stats["item_rows"][item]), fmt_int(len(stats["item_payapps"].get(item, set())))]
        for item, amount in sorted(stats["item_total_billed"].items(), key=lambda item: item[1], reverse=True)[:20]
    ]
    prefix_rows = [
        [prefix, fmt_int(stats["standard_prefix"][prefix]), fmt_dec(amount, 2)]
        for prefix, amount in sorted(stats["standard_prefix_by_amount"].items(), key=lambda item: item[1], reverse=True)[:25]
    ]

    cells = [
        markdown_cell(
            "# Customer Payment Details EDA\n\n"
            "This notebook performs an exploratory analysis of `custpaydetails.csv.csv`, which was exported from `custpaydetails.sql`. "
            "The analysis focuses on field-by-field representative data, ranges, missingness patterns, date semantics, and basic cross-field consistency checks."
        ),
        markdown_cell(
            "## Executive Summary\n\n"
            + findings["overview"]
            + "\n\nPrimary interpretation: each row is a UDOT contract pay item joined to a work posting, pay estimate, and pay estimate detail row. "
            "The result is not one row per project, contract, item, or pay application; the grain is closer to a paid posting/detail line for a contract item."
        ),
        markdown_cell(
            "## SQL-Derived Context\n\n"
            "The active query defines a `udot_contract_payment_details` CTE and selects from it. Other customer blocks are present but commented out. "
            "The CTE starts with `CORITEMItemDetails` contract items, joins contracts and projects as required tables, then left-joins lookup tables, recursive container paths, linked budget items, commitments, work postings, pay estimate details, and pay estimates. "
            "The `WHERE` clause requires `WP.STATUS IS NOT NULL`, `PE.STATUS IS NOT NULL`, and `ContractEndDate < CURRENT_DATE`.\n\n"
            + md_table(["UDOT source table referenced"], [[table] for table in active_tables])
        ),
        code_cell(
            "import csv\n"
            "from collections import Counter, defaultdict\n"
            "from datetime import datetime\n"
            "from decimal import Decimal, InvalidOperation\n"
            "from pathlib import Path\n\n"
            "CSV_PATH = Path('custpaydetails.csv.csv')\n"
            "with CSV_PATH.open(newline='', encoding='utf-8-sig') as handle:\n"
            "    reader = csv.DictReader(handle)\n"
            "    rows = list(reader)\n"
            "    fields = reader.fieldnames\n\n"
            "print(f'Rows: {len(rows):,}')\n"
            "print(f'Fields: {len(fields):,}')\n"
            "print(fields)\n"
        ),
        markdown_cell("## Field Inventory\n\n" + md_table(
            [
                "Field",
                "Inferred role",
                "Missing",
                "Missing %",
                "Unique",
                "Representative examples",
                "Most common values",
                "SQL/source meaning",
                "Missingness interpretation",
            ],
            field_rows,
        )),
        markdown_cell("## Missingness Ranked\n\n" + md_table(["Field", "Missing", "Missing %", "Unique non-missing"], missing_rows)),
        markdown_cell("## Date Fields and Ranges\n\n" + md_table(
            ["Field", "Parsed", "Missing", "Parse failures", "Min", "Q1-ish", "Median-ish", "Q3-ish", "Max", "Interpretation"],
            date_rows,
        )),
        markdown_cell(
            "## Date Relationship Checks\n\n"
            "These checks help separate business semantics from data quality issues. `WPPOSTINGDATE` and `POSTINGDATE` are the same selected source column in the SQL, so they should match exactly. "
            "`FROMDATE` and `TODATE` describe the pay estimate period, which may not perfectly contain the posting date depending on operational practice.\n\n"
            + md_table(["Check", "OK rows", "Exception rows", "Exception %"], date_check_rows)
            + "\n\n### Durations and Date Differences\n\n"
            + md_table(["Difference", "Rows", "Min days", "Median days", "Max days", "Mean days"], duration_rows)
        ),
        markdown_cell("## Numeric Fields and Ranges\n\n" + md_table(
            ["Field", "Parsed", "Missing", "Parse failures", "Min", "P05", "P25", "Median", "P75", "P95", "Max", "Sum"],
            numeric_rows,
        )),
        markdown_cell(
            "## Amount Consistency Checks\n\n"
            "The SQL computes `TOTALPOSTINGAMOUNT` as `POSTINGQTY * POSTINGUNITRATE` and `TOTALWORKCOMPLETED` as `ITEMUNITPRICE * QTYBILLEDTHISPAYAPP`. "
            "`BILLEDAMOUNT` comes from pay estimate details and is expected to line up with `BILLINGRATE * QTYBILLEDTHISPAYAPP` when rates are populated consistently.\n\n"
            + md_table(["Check", "Matches", "Mismatches", "Mismatch %"], amount_check_rows)
        ),
        markdown_cell("## Categorical and Text Fields\n\n" + md_table(
            ["Field", "Missing", "Missing %", "Unique", "Top values"],
            categorical_rows,
        )),
        markdown_cell(
            "## Grain and Entity Counts\n\n"
            + md_table(
                ["Metric", "Value"],
                [
                    ["Rows", fmt_int(n)],
                    ["Unique projects", fmt_int(len(stats["unique"].get("PROJECTCODE", [])))],
                    ["Unique contracts", fmt_int(len(stats["unique"].get("CONTRACTID", [])))],
                    ["Unique contract items", fmt_int(len(stats["unique"].get("ITEMID", [])))],
                    ["Unique postings", fmt_int(len(stats["unique"].get("POSTINGID", [])))],
                    ["Unique pay apps", fmt_int(len(stats["unique"].get("PAYAPPID", [])))],
                    ["Unique pay estimate detail rows", fmt_int(len(stats["unique"].get("PEDETAILSID", [])))],
                    ["Unique item/pay-app pairs", fmt_int(len({(r.get('ITEMID'), r.get('PAYAPPID')) for r in rows}))],
                ],
            )
            + "\n\n### Contracts With Most Distinct Items\n\n"
            + md_table(["CONTRACTID", "Distinct ITEMID count"], top_contract_rows)
            + "\n\n### Number of Pay Apps per Contract Item\n\n"
            + md_table(["Distinct pay apps on item", "Item count", "Share of items"], payapp_dist_rows)
        ),
        markdown_cell(
            "## High-Value and Standard Item Patterns\n\n"
            "These tables are useful for deciding which item categories or large-dollar lines deserve follow-up modeling or manual review.\n\n"
            "### Highest Total Billed Items\n\n"
            + md_table(["ITEMID", "Total billed amount", "Rows", "Distinct pay apps"], top_item_amount_rows)
            + "\n\n### Standard Item Number Prefixes by Billed Amount\n\n"
            + md_table(["STANDARDITEMNO prefix", "Rows", "Total billed amount"], prefix_rows)
        ),
        markdown_cell(
            "## Practical Interpretation of Dates\n\n"
            "- `CONTRACTSTARTDATE` and `CONTRACTENDDATE` come from the contract master. Because the SQL filters closed contracts with `ContractEndDate < CURRENT_DATE`, this dataset intentionally emphasizes completed contracts.\n"
            "- `PROJSTARTDATE` and `PROJENDDATE` come from the project master. They are broader project dates and can differ from contract dates.\n"
            "- `PHASEITEMSTARTDATE` and `PHASEITEMENDDATE` are inherited through the contract-item container hierarchy. Missing values usually mean the item container is not tied to a phase item with dates, not that the contract item itself is missing.\n"
            "- `WPPOSTINGDATE` and `POSTINGDATE` are both selected from `PROCMGTWorkPosting.POSTINGDATE`; the duplicate fields are redundant in this export.\n"
            "- `FROMDATE` and `TODATE` are pay estimate period boundaries. They are repeated across many item/detail rows sharing the same `PAYAPPID`.\n"
        ),
        markdown_cell(
            "## Follow-Up Analysis Ideas\n\n"
            "- Aggregate to one row per `ITEMID` for production curves, because the current row grain repeats item attributes across pay app/detail rows.\n"
            "- Aggregate to one row per `PAYAPPID` to study pay estimate cycles and period lengths.\n"
            "- Investigate linked budget missingness separately. Budget fields are left-joined and are structurally optional; missingness here does not necessarily imply a broken payment record.\n"
            "- Review rows where amount checks mismatch or date checks fail before using the data for forecasting or training."
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    fields, rows = load_rows()
    stats = profile_data(fields, rows)
    notebook = build_notebook(fields, rows, stats)
    OUT_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
