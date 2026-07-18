from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv")
DICT_PATH = Path("project_feature_non_keyword_field_dictionary_2026-06-10.csv")
KEYWORD_DETAIL_PATH = Path("approved_keyword_feature_column_filter_detail_2026-06-11.csv")

PRIMARY_OUT = Path("approved_keyword_feature_spearman_correlations_with_sources_2026-06-15.csv")
EXCLUDED_OUT = Path("approved_keyword_feature_spearman_correlations_excluded_features_2026-06-15.csv")
SUMMARY_OUT = Path("approved_keyword_feature_spearman_correlations_with_sources_2026-06-15.md")

TARGET = "PERCENTDELAYED"


TARGET_OR_LEAKAGE_PREFIXES = ("TARGET",)
IDENTITY_OR_DESCRIPTOR_FIELDS = {
    "RECORD_ID",
    "CUSTOMERNAME",
    "PROJECTID",
    "PROJECTNAME",
    "PROJECTCODE",
    "PROJECTDESCRIPTION",
    "PROJECTSTATUS",
}


SOURCE_BASE_NOTES = {
    "PROJECT": "Tokenized from PROJECTPROJECTMAIN.DESCRIPTION after lower-case text normalization.",
    "CONTRACT": "Tokenized from CONTMGTMASTER.NAME and CONTMGTMASTER.DESC after lower-case text normalization.",
    "ITEM": "Tokenized from CORITEMITEMDETAILS.DESCRIPTION after lower-case text normalization.",
}


def canonical(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def remove_prefix_suffix(value: str, prefix: str, suffix: str) -> str:
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value


def read_keyword_map() -> dict[str, str]:
    if not KEYWORD_DETAIL_PATH.exists():
        return {}
    detail = pd.read_csv(KEYWORD_DETAIL_PATH)
    if not {"column", "keyword", "retained_in_output"}.issubset(detail.columns):
        return {}
    retained = detail[detail["retained_in_output"].astype(bool)]
    return {
        str(row["column"]).upper(): str(row["keyword"]).replace("'", "''")
        for _, row in retained.iterrows()
        if pd.notna(row["keyword"])
    }


def keyword_parts(feature: str, keyword_map: dict[str, str]) -> tuple[str | None, str | None, str | None]:
    f = feature.upper()
    keyword = keyword_map.get(f)
    if f.startswith("PROJ_KW_") and f.endswith("_COUNT"):
        raw = remove_prefix_suffix(f, "PROJ_KW_", "_COUNT")
        return "project_keyword_count", keyword or raw.lower(), "project"
    if f.startswith("CONTRACT_KW_") and f.endswith("_CONTRACT_COUNT"):
        raw = remove_prefix_suffix(f, "CONTRACT_KW_", "_CONTRACT_COUNT")
        return "contract_keyword_count", keyword or raw.lower(), "contract"
    if f.startswith("CONTRACT_KW_") and f.endswith("_CONTRACT_SHARE"):
        raw = remove_prefix_suffix(f, "CONTRACT_KW_", "_CONTRACT_SHARE")
        return "contract_keyword_share", keyword or raw.lower(), "contract"
    if f.startswith("ITEM_KW_") and f.endswith("_ITEM_COUNT"):
        raw = remove_prefix_suffix(f, "ITEM_KW_", "_ITEM_COUNT")
        return "item_keyword_count", keyword or raw.lower(), "item"
    if f.startswith("ITEM_KW_") and f.endswith("_ITEM_SHARE"):
        raw = remove_prefix_suffix(f, "ITEM_KW_", "_ITEM_SHARE")
        return "item_keyword_share", keyword or raw.lower(), "item"
    if f.startswith("ITEM_KW_") and f.endswith("_PLANNED_VALUE_SHARE"):
        raw = remove_prefix_suffix(f, "ITEM_KW_", "_PLANNED_VALUE_SHARE")
        return "item_keyword_planned_value_share", keyword or raw.lower(), "item"
    return None, None, None


def keyword_source(feature: str, keyword_map: dict[str, str]) -> tuple[str, str, str]:
    klass, keyword, family = keyword_parts(feature, keyword_map)
    if klass is None or keyword is None or family is None:
        return "unclassified_keyword", "", ""

    if klass == "project_keyword_count":
        source = f"COALESCE(SUM(IFF(project_token_rows.Keyword = '{keyword}', 1, 0)), 0)"
        detail = SOURCE_BASE_NOTES["PROJECT"]
    elif klass == "contract_keyword_count":
        source = (
            f"COALESCE(SUM(IFF(contract_keyword_counts.Keyword = '{keyword}', "
            "contract_keyword_counts.ContractKeywordContractCount, 0)), 0)"
        )
        detail = SOURCE_BASE_NOTES["CONTRACT"]
    elif klass == "contract_keyword_share":
        source = (
            f"COALESCE(SUM(IFF(contract_keyword_counts.Keyword = '{keyword}', "
            "contract_keyword_counts.ContractKeywordContractCount, 0)), 0) "
            "/ NULLIF(MAX(project_nontext_features.NumContracts), 0)"
        )
        detail = SOURCE_BASE_NOTES["CONTRACT"]
    elif klass == "item_keyword_count":
        source = (
            f"COALESCE(SUM(IFF(item_keyword_counts.Keyword = '{keyword}', "
            "item_keyword_counts.ItemKeywordItemCount, 0)), 0)"
        )
        detail = SOURCE_BASE_NOTES["ITEM"]
    elif klass == "item_keyword_share":
        source = (
            f"COALESCE(SUM(IFF(item_keyword_counts.Keyword = '{keyword}', "
            "item_keyword_counts.ItemKeywordItemCount, 0)), 0) "
            "/ NULLIF(MAX(project_nontext_features.NumContractItems), 0)"
        )
        detail = SOURCE_BASE_NOTES["ITEM"]
    else:
        source = (
            f"COALESCE(SUM(IFF(item_keyword_counts.Keyword = '{keyword}', "
            "item_keyword_counts.ItemKeywordAbsPlannedValue, 0)), 0) "
            "/ NULLIF(MAX(project_nontext_features.AbsProjectPlannedValue), 0)"
        )
        detail = (
            SOURCE_BASE_NOTES["ITEM"]
            + " Item planned value is ABS(CORITEMITEMDETAILS.UnitPrice * CORITEMITEMDETAILS.ContractQuantity)."
        )
    return klass, source, detail


def load_dictionary() -> dict[str, dict[str, str]]:
    if not DICT_PATH.exists():
        return {}
    d = pd.read_csv(DICT_PATH).fillna("")
    records: dict[str, dict[str, str]] = {}
    for _, row in d.iterrows():
        records[canonical(row["FieldName"])] = {
            "class": str(row.get("FieldGroup", "")).strip() or "non_keyword",
            "source": str(row.get("CalculationDerivation", "")).strip(),
            "source_detail": str(row.get("GeneralDescription", "")).strip(),
            "beginning_available": str(row.get("BeginningAvailable", "")).strip(),
        }
    return records


def classify_source(feature: str, dictionary: dict[str, dict[str, str]], keyword_map: dict[str, str]) -> tuple[str, str, str, str]:
    if "_KW_" in feature:
        klass, source, source_detail = keyword_source(feature, keyword_map)
        return klass, source, source_detail, "Yes"

    entry = dictionary.get(canonical(feature), {})
    klass = entry.get("class", "non_keyword")
    source = entry.get("source", "")
    source_detail = entry.get("source_detail", "")
    beginning_available = entry.get("beginning_available", "")
    return klass, source, source_detail, beginning_available


def exclusion_reason(feature: str, series: pd.Series) -> str:
    f = feature.upper()
    if f == TARGET:
        return "target field"
    if f.startswith(TARGET_OR_LEAKAGE_PREFIXES):
        return "retrospective target/leakage field"
    if f in IDENTITY_OR_DESCRIPTOR_FIELDS:
        return "identity or descriptor field excluded from primary correlation table"
    if not pd.api.types.is_numeric_dtype(series):
        return "non-numeric field; Spearman correlation not computed"
    return ""


def spearman_stats(feature: str, series: pd.Series, target: pd.Series) -> dict[str, float | int | str | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    paired = pd.concat([numeric, target], axis=1).dropna()
    n_pairs = int(len(paired))
    non_null = int(numeric.notna().sum())
    missing_rate = float(1 - non_null / len(numeric)) if len(numeric) else math.nan
    unique_non_null = int(numeric.dropna().nunique())
    paired_unique = int(paired.iloc[:, 0].nunique()) if n_pairs else 0
    target_unique = int(paired.iloc[:, 1].nunique()) if n_pairs else 0

    if n_pairs < 3 or paired_unique < 2 or target_unique < 2:
        rho = np.nan
    else:
        rho = float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman"))

    if pd.isna(rho):
        direction = "zero_or_undefined"
    elif rho > 0:
        direction = "positive"
    elif rho < 0:
        direction = "negative"
    else:
        direction = "zero_or_undefined"

    return {
        "feature": feature,
        "spearman_r": rho,
        "direction": direction,
        "non_null_pairs": n_pairs,
        "feature_non_null_count": non_null,
        "feature_missing_rate": missing_rate,
        "feature_unique_non_null_values": unique_non_null,
    }



def simple_markdown_table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "No rows."
    values = frame[cols].copy()
    for col in values.columns:
        if pd.api.types.is_float_dtype(values[col]):
            values[col] = values[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
        else:
            values[col] = values[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values.astype(str).values.tolist()]
    return "\n".join([header, sep] + body)

def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["direction_rank"] = pd.NA
    pos_idx = frame[frame["direction"].eq("positive")].sort_values(
        ["spearman_r", "feature"], ascending=[False, True]
    ).index
    neg_idx = frame[frame["direction"].eq("negative")].sort_values(
        ["spearman_r", "feature"], ascending=[True, True]
    ).index
    frame.loc[pos_idx, "direction_rank"] = range(1, len(pos_idx) + 1)
    frame.loc[neg_idx, "direction_rank"] = range(1, len(neg_idx) + 1)

    frame["absolute_rank"] = pd.NA
    valid_idx = frame[frame["spearman_r"].notna()].assign(
        abs_r=lambda x: x["spearman_r"].abs()
    ).sort_values(["abs_r", "feature"], ascending=[False, True]).index
    frame.loc[valid_idx, "absolute_rank"] = range(1, len(valid_idx) + 1)

    direction_order = {"positive": 0, "negative": 1, "zero_or_undefined": 2}
    frame["_direction_order"] = frame["direction"].map(direction_order).fillna(3)
    frame["_sort_rank"] = frame["direction_rank"].fillna(10**9).astype(int)
    frame = frame.sort_values(["_direction_order", "_sort_rank", "feature"]).drop(
        columns=["_direction_order", "_sort_rank"]
    )
    return frame


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET} in {DATA_PATH}")

    target = pd.to_numeric(df[TARGET], errors="coerce")
    dictionary = load_dictionary()
    keyword_map = read_keyword_map()

    rows = []
    excluded_rows = []
    for feature in df.columns:
        series = df[feature]
        reason = exclusion_reason(feature, series)
        stats = spearman_stats(feature, series, target) if pd.api.types.is_numeric_dtype(series) else {
            "feature": feature,
            "spearman_r": np.nan,
            "direction": "zero_or_undefined",
            "non_null_pairs": int(series.notna().sum()),
            "feature_non_null_count": int(series.notna().sum()),
            "feature_missing_rate": float(1 - series.notna().sum() / len(series)),
            "feature_unique_non_null_values": int(series.dropna().nunique()),
        }
        klass, source, source_detail, beginning_available = classify_source(feature, dictionary, keyword_map)
        stats.update(
            {
                "class": klass,
                "source": source,
                "source_detail": source_detail,
                "beginning_available": beginning_available,
                "excluded_reason": reason,
            }
        )

        if reason:
            excluded_rows.append(stats)
        else:
            rows.append(stats)

    primary = add_ranks(pd.DataFrame(rows))
    excluded = add_ranks(pd.DataFrame(excluded_rows))

    output_cols = [
        "direction",
        "direction_rank",
        "absolute_rank",
        "feature",
        "spearman_r",
        "class",
        "source",
        "source_detail",
        "beginning_available",
        "non_null_pairs",
        "feature_non_null_count",
        "feature_missing_rate",
        "feature_unique_non_null_values",
        "excluded_reason",
    ]
    primary[output_cols].to_csv(PRIMARY_OUT, index=False)
    excluded[output_cols].to_csv(EXCLUDED_OUT, index=False)

    positive = int((primary["direction"] == "positive").sum())
    negative = int((primary["direction"] == "negative").sum())
    undefined = int((primary["direction"] == "zero_or_undefined").sum())
    top_pos = primary[primary["direction"].eq("positive")].head(10)
    top_neg = primary[primary["direction"].eq("negative")].head(10)

    lines = [
        "# Approved Keyword Feature Correlation Source Mapping",
        "",
        f"Dataset: `{DATA_PATH.name}`",
        f"Target: `{TARGET}`",
        "",
        "This extract ranks non-leakage numeric features by Spearman correlation against project percent delay. "
        "Positive and negative features are ranked separately, matching the prior top-correlates notebook convention. "
        "An absolute rank is also included for a single overall strength ordering.",
        "",
        "Excluded fields were written to a separate CSV with the same schema. Exclusions include the target itself, "
        "retrospective target/leakage fields, identity/descriptive fields, and nonnumeric fields where Spearman "
        "correlation is not meaningful.",
        "",
        "## Counts",
        "",
        f"- Primary ranked features: {len(primary):,}",
        f"- Positive Spearman features: {positive:,}",
        f"- Negative Spearman features: {negative:,}",
        f"- Zero/undefined Spearman features: {undefined:,}",
        f"- Excluded fields: {len(excluded):,}",
        "",
        "## Top Positive Features",
        "",
        simple_markdown_table(top_pos, ["direction_rank", "feature", "spearman_r", "class"]),
        "",
        "## Top Negative Features",
        "",
        simple_markdown_table(top_neg, ["direction_rank", "feature", "spearman_r", "class"]),
        "",
        "## Source Mapping Notes",
        "",
        "- Non-keyword features use the derivation documented in the project feature field dictionary.",
        "- Project keyword features are token counts from `PROJECTPROJECTMAIN.DESCRIPTION`.",
        "- Contract keyword features are token counts or shares from `CONTMGTMASTER.NAME` and `CONTMGTMASTER.DESC`.",
        "- Item keyword features are token counts, item shares, or planned-value shares from `CORITEMITEMDETAILS.DESCRIPTION` and planned item values.",
        "",
        "## Outputs",
        "",
        f"- Primary table: `{PRIMARY_OUT.name}`",
        f"- Excluded table: `{EXCLUDED_OUT.name}`",
    ]
    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {PRIMARY_OUT} ({len(primary):,} rows)")
    print(f"Wrote {EXCLUDED_OUT} ({len(excluded):,} rows)")
    print(f"Wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
