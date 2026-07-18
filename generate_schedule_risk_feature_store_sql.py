from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


SOURCE = Path("custpaydetails_project_feature_table_with_keywords_materialized.sql")
MANIFEST = Path("approved_keyword_feature_column_filter_detail_2026-06-11.csv")
OUTPUT = Path("schedule_risk_feature_calculation.sql")


def replace_keyword_list(sql: str, family: str, keywords: list[str], next_marker: str) -> str:
    start_marker = f"CREATE OR REPLACE TEMPORARY TABLE {family}_keyword_list AS"
    start = sql.index(start_marker)
    end = sql.index(next_marker, start)
    values = ",\n".join(f"        ('{keyword.replace(chr(39), chr(39) * 2)}')" for keyword in keywords)
    block = (
        f"CREATE OR REPLACE TEMPORARY TABLE {family}_keyword_list AS\n"
        "SELECT column1::STRING AS Keyword\n"
        "    FROM VALUES\n"
        f"{values}\n"
        ";\n\n"
    )
    return sql[:start] + block + sql[end:]


def filter_feature_block(sql: str, table: str, prefix: str, approved: set[str], next_marker: str) -> str:
    start_marker = f"CREATE OR REPLACE TEMPORARY TABLE {table} AS"
    start = sql.index(start_marker)
    end = sql.index(next_marker, start)
    block = sql[start:end]
    kept: list[str] = []
    for line in block.splitlines():
        match = re.search(r"\bAS\s+([A-Z0-9_]+),?\s*$", line)
        if match and match.group(1).startswith(prefix) and match.group(1) not in approved:
            continue
        kept.append(line)
    feature_indexes = []
    for index, line in enumerate(kept):
        match = re.search(r"\bAS\s+([A-Z0-9_]+),?\s*$", line)
        if match and match.group(1).startswith(prefix):
            feature_indexes.append(index)
    for index in feature_indexes[:-1]:
        if not kept[index].rstrip().endswith(","):
            kept[index] = kept[index].rstrip() + ","
    if feature_indexes:
        last_index = feature_indexes[-1]
        kept[last_index] = kept[last_index].rstrip().rstrip(",")
    return sql[:start] + "\n".join(kept) + "\n" + sql[end:]


def main() -> None:
    sql = SOURCE.read_text(encoding="utf-8")
    manifest = pd.read_csv(MANIFEST)
    retained = manifest[manifest["retained_in_output"].astype(bool)].copy()
    approved = set(retained["column"].astype(str))

    # The inference feature calculation must not touch payment-derived targets.
    payment_start = sql.index("CREATE OR REPLACE TEMPORARY TABLE payment_rows AS")
    setup_start = sql.index("CREATE OR REPLACE TEMPORARY TABLE project_text_entities AS")
    sql = sql[:payment_start] + sql[setup_start:]

    project_keywords = sorted(retained.loc[retained["family"].eq("project"), "keyword"].dropna().astype(str).unique())
    contract_keywords = sorted(retained.loc[retained["family"].eq("contract"), "keyword"].dropna().astype(str).unique())
    item_keywords = sorted(retained.loc[retained["family"].eq("item"), "keyword"].dropna().astype(str).unique())

    sql = replace_keyword_list(sql, "project", project_keywords, "-- Optional sanity check while tuning: SELECT COUNT(*) AS project_keyword_list_rows")
    sql = replace_keyword_list(sql, "contract", contract_keywords, "-- Optional sanity check while tuning: SELECT COUNT(*) AS contract_keyword_list_rows")
    sql = replace_keyword_list(sql, "item", item_keywords, "-- Optional sanity check while tuning: SELECT COUNT(*) AS item_keyword_list_rows")

    sql = filter_feature_block(
        sql,
        "project_keyword_features",
        "PROJ_KW_",
        approved,
        "-- Optional sanity check while tuning: SELECT COUNT(*) AS project_keyword_features_rows",
    )
    sql = filter_feature_block(
        sql,
        "contract_keyword_features",
        "CONTRACT_KW_",
        approved,
        "-- Optional sanity check while tuning: SELECT COUNT(*) AS contract_keyword_features_rows",
    )
    sql = filter_feature_block(
        sql,
        "item_keyword_features",
        "ITEM_KW_",
        approved,
        "-- Optional sanity check while tuning: SELECT COUNT(*) AS item_keyword_features_rows",
    )

    # Remove fields whose current values are not guaranteed to be setup-time values.
    sql = sql.replace(
        "COALESCE(p.ProjectName, '') || ' ' || COALESCE(p.ProjectDescription, '') || ' ' || COALESCE(p.ProjectStatus, '')",
        "COALESCE(p.ProjectName, '') || ' ' || COALESCE(p.ProjectDescription, '')",
    )
    sql = sql.replace(
        "COALESCE(i.ItemDescription, '') || ' ' || COALESCE(i.BudgetItemDescription, '')",
        "COALESCE(i.ItemDescription, '')",
    )

    final_start = sql.index("CREATE OR REPLACE TEMPORARY TABLE project_feature_table_with_keywords_final AS")
    final_block = """CREATE OR REPLACE TEMPORARY TABLE schedule_project_features_build AS
SELECT
    n.*,
    pk.* EXCLUDE (CustomerName, ProjectID),
    ck.* EXCLUDE (CustomerName, ProjectID),
    ik.* EXCLUDE (CustomerName, ProjectID),
    CURRENT_TIMESTAMP()::TIMESTAMP_TZ AS FeatureAsOfUtc,
    'schedule-project-features-v1'::STRING AS FeatureSchemaVersion,
    'approved-keywords-2026-06-11'::STRING AS KeywordManifestVersion
FROM project_nontext_features n
LEFT JOIN project_keyword_features pk
    ON pk.CustomerName = n.CustomerName
   AND pk.ProjectID = n.ProjectID
LEFT JOIN contract_keyword_features ck
    ON ck.CustomerName = n.CustomerName
   AND ck.ProjectID = n.ProjectID
LEFT JOIN item_keyword_features ik
    ON ik.CustomerName = n.CustomerName
   AND ik.ProjectID = n.ProjectID
ORDER BY n.CustomerName, n.ProjectID
;

-- The refresh script publishes this temporary build table atomically.
SELECT COUNT(*) AS BuildRowCount FROM schedule_project_features_build;
"""
    sql = sql[:final_start] + final_block

    header = """-- Schedule Risk Agent: approved beginning-feature calculation
-- Generated from custpaydetails_project_feature_table_with_keywords_materialized.sql
-- and approved_keyword_feature_column_filter_detail_2026-06-11.csv.
--
-- This script intentionally excludes payment rows, delay targets, change orders,
-- current project status from tokenization, and linked budget-item descriptions
-- from item keyword tokenization. Retrain the production model on this exact schema.
-- Run in one Snowflake session before schedule_risk_feature_store_refresh_current.sql.

"""
    OUTPUT.write_text(header + sql, encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(project_keywords)} project, {len(contract_keywords)} contract, and {len(item_keywords)} item keywords")


if __name__ == "__main__":
    main()
