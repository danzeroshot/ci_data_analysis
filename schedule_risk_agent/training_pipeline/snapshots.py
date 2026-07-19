from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (
    DataContractError,
    KEY_COLUMNS,
    LABEL_COLUMN,
    TARGET_COLUMN,
)
from .lineage import (
    compact_utc_now,
    sha256_file,
    utc_now,
    verify_checksums,
    write_checksums,
    write_json_atomic,
)


def normalize_project_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text


def normalize_customer(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["CUSTOMERNAME"] = frame["CUSTOMERNAME"].map(normalize_customer)
    frame["PROJECTID"] = frame["PROJECTID"].map(normalize_project_id)
    return frame


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataContractError("Invalid snapshot manifest {}: {}".format(path, exc)) from exc


def verify_snapshot(data_path: Path, manifest_path: Path, require_complete: bool = True) -> Dict[str, Any]:
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    root = manifest_path.parent
    if require_complete and not (root / "COMPLETE").exists():
        raise DataContractError("Snapshot is not complete: {}".format(root))
    checksums_path = root / "checksums.sha256"
    if checksums_path.exists():
        verify_checksums(root, checksums_path)
    if not data_path.is_file():
        raise DataContractError("Snapshot data file does not exist: {}".format(data_path))
    expected_rows = manifest.get("row_count")
    if expected_rows is not None:
        actual_rows = pq.ParquetFile(data_path).metadata.num_rows
        if int(expected_rows) != int(actual_rows):
            raise DataContractError(
                "Snapshot row count mismatch: expected {}, found {}".format(
                    expected_rows, actual_rows
                )
            )
    return manifest


def load_feature_snapshot(data_path: Path, manifest_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    manifest = verify_snapshot(data_path, manifest_path)
    frame = pd.read_parquet(data_path)
    missing = [column for column in KEY_COLUMNS if column not in frame]
    if missing:
        raise DataContractError("Feature snapshot missing keys: " + ", ".join(missing))
    frame.columns = [str(column).upper() for column in frame.columns]
    frame = normalize_keys(frame)
    duplicates = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise DataContractError(
            "Feature snapshot contains {} duplicate project rows".format(int(duplicates.sum()))
        )
    return frame, manifest


def load_label_snapshot(data_path: Path, manifest_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    manifest = verify_snapshot(data_path, manifest_path)
    frame = pd.read_parquet(data_path)
    frame.columns = [str(column).upper() for column in frame.columns]
    required = list(KEY_COLUMNS) + [TARGET_COLUMN, LABEL_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise DataContractError("Label snapshot missing fields: " + ", ".join(missing))
    frame = normalize_keys(frame)
    duplicates = frame.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise DataContractError(
            "Label snapshot contains {} duplicate project rows".format(int(duplicates.sum()))
        )
    invalid_labels = ~frame[LABEL_COLUMN].isin([0, 1, 2])
    if invalid_labels.any():
        raise DataContractError("Label snapshot contains labels outside 0, 1, and 2")
    return frame, manifest


def create_label_snapshot_from_csv(
    source_csv: Path,
    output_root: Path,
    target_version: str,
) -> Path:
    source_csv = Path(source_csv)
    source_hash = sha256_file(source_csv)
    build_id = "schedule-labels-{}-{}".format(compact_utc_now(), source_hash[:8])
    final_dir = Path(output_root) / build_id
    staging = Path(output_root) / ("." + build_id + ".staging")
    if final_dir.exists() or staging.exists():
        raise DataContractError("Label snapshot build already exists: {}".format(build_id))
    staging.mkdir(parents=True)
    frame = pd.read_csv(source_csv, low_memory=False)
    frame.columns = [str(column).upper() for column in frame.columns]
    required = list(KEY_COLUMNS) + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise DataContractError("Source CSV missing label fields: " + ", ".join(missing))
    frame = normalize_keys(frame)
    values = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid = frame[list(KEY_COLUMNS)].copy()
    valid[TARGET_COLUMN] = values
    valid["EXCLUSIONREASON"] = np.where(
        valid["CUSTOMERNAME"].eq("") | valid["PROJECTID"].eq(""),
        "missing_project_key",
        np.where(values.isna(), "missing_or_non_numeric_percent_delayed", ""),
    )
    duplicate_mask = valid.duplicated(list(KEY_COLUMNS), keep=False)
    valid.loc[duplicate_mask, "EXCLUSIONREASON"] = "duplicate_project_key"
    exclusions = valid[valid["EXCLUSIONREASON"].ne("")].copy()
    labels = valid[valid["EXCLUSIONREASON"].eq("")].drop(columns=["EXCLUSIONREASON"])
    labels[LABEL_COLUMN] = pd.cut(
        labels[TARGET_COLUMN],
        [-np.inf, 0.0, 25.0, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype(int)
    pq.write_table(
        pa.Table.from_pandas(labels, preserve_index=False),
        staging / "labels.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(exclusions, preserve_index=False),
        staging / "exclusions.parquet",
        compression="zstd",
    )
    class_counts = {
        str(int(key)): int(value)
        for key, value in labels[LABEL_COLUMN].value_counts().sort_index().items()
    }
    manifest = {
        "schema_version": "schedule-label-snapshot-v1",
        "build_id": build_id,
        "source_type": "legacy_csv",
        "development_only": True,
        "source_path": str(source_csv),
        "source_sha256": source_hash,
        "target_definition_version": target_version,
        "created_at_utc": utc_now(),
        "row_count": int(len(labels)),
        "excluded_row_count": int(len(exclusions)),
        "class_counts": class_counts,
        "customers": sorted(labels["CUSTOMERNAME"].unique().tolist()),
        "labels_file": "labels.parquet",
    }
    write_json_atomic(staging / "manifest.json", manifest)
    profile = {
        "percent_delayed": {
            "minimum": float(labels[TARGET_COLUMN].min()),
            "maximum": float(labels[TARGET_COLUMN].max()),
            "quantiles": {
                str(q): float(labels[TARGET_COLUMN].quantile(q))
                for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
            },
        },
        "class_counts": class_counts,
        "exclusion_counts": exclusions["EXCLUSIONREASON"].value_counts().to_dict(),
    }
    write_json_atomic(staging / "profile.json", profile)
    write_checksums(staging, [
        "labels.parquet", "exclusions.parquet", "manifest.json", "profile.json"
    ])
    (staging / "COMPLETE").write_text("", encoding="ascii")
    staging.rename(final_dir)
    return final_dir


def join_training_data(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    feature_keys = features[list(KEY_COLUMNS)]
    label_keys = labels[list(KEY_COLUMNS)]
    merged = features.merge(
        labels[list(KEY_COLUMNS) + [TARGET_COLUMN, LABEL_COLUMN]],
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    feature_only = feature_keys.merge(label_keys, on=list(KEY_COLUMNS), how="left", indicator=True)
    label_only = label_keys.merge(feature_keys, on=list(KEY_COLUMNS), how="left", indicator=True)
    reconciliation = {
        "feature_rows": int(len(features)),
        "label_rows": int(len(labels)),
        "matched_rows": int(len(merged)),
        "feature_only_rows": int(feature_only["_merge"].eq("left_only").sum()),
        "label_only_rows": int(label_only["_merge"].eq("left_only").sum()),
        "match_rate_from_features": float(len(merged) / len(features)) if len(features) else None,
        "match_rate_from_labels": float(len(merged) / len(labels)) if len(labels) else None,
        "class_counts": {
            str(int(key)): int(value)
            for key, value in merged[LABEL_COLUMN].value_counts().sort_index().items()
        },
        "customer_rows": merged["CUSTOMERNAME"].value_counts().sort_index().to_dict(),
    }
    if not len(merged):
        raise DataContractError("Feature and label snapshots have no matching project keys")
    return merged, reconciliation

def create_label_snapshot_from_snowflake(
    sql_path: Path,
    output_root: Path,
    target_version: str,
) -> Path:
    from io import StringIO

    from schedule_risk_agent.config import Settings
    from schedule_risk_agent.snowflake_access import connect

    sql_path = Path(sql_path)
    sql = sql_path.read_text(encoding="utf-8")
    sql_hash = hashlib.sha256(sql.encode("utf-8")).hexdigest()
    settings = Settings.load()
    connection = connect(settings)
    final_cursor = None
    statement_ids = []
    try:
        for cursor in connection.execute_stream(StringIO(sql)):
            final_cursor = cursor
            statement_ids.append(cursor.sfqid)
        if final_cursor is None:
            raise DataContractError("Label SQL executed no statements")
        frame = final_cursor.fetch_pandas_all()
    finally:
        connection.close()

    frame.columns = [str(column).upper() for column in frame.columns]
    required = list(KEY_COLUMNS) + [TARGET_COLUMN, LABEL_COLUMN]
    missing = [column for column in required if column not in frame]
    if missing:
        raise DataContractError("Snowflake label query missing fields: " + ", ".join(missing))
    frame = normalize_keys(frame)
    exclusion_column = "LABELEXCLUSIONREASON"
    if exclusion_column not in frame:
        frame[exclusion_column] = None
    duplicate_mask = frame.duplicated(list(KEY_COLUMNS), keep=False)
    frame.loc[duplicate_mask, exclusion_column] = "duplicate_project_key"
    valid_mask = (
        frame[exclusion_column].isna()
        & frame[TARGET_COLUMN].notna()
        & frame[LABEL_COLUMN].isin([0, 1, 2])
        & frame["CUSTOMERNAME"].ne("")
        & frame["PROJECTID"].ne("")
    )
    labels = frame.loc[valid_mask].copy()
    exclusions = frame.loc[~valid_mask].copy()
    labels[LABEL_COLUMN] = labels[LABEL_COLUMN].astype(int)

    build_id = "schedule-labels-{}-{}".format(compact_utc_now(), sql_hash[:8])
    final_dir = Path(output_root) / build_id
    staging = Path(output_root) / ("." + build_id + ".staging")
    if final_dir.exists() or staging.exists():
        raise DataContractError("Label snapshot build already exists: {}".format(build_id))
    staging.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pandas(labels, preserve_index=False),
        staging / "labels.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pandas(exclusions, preserve_index=False),
        staging / "exclusions.parquet",
        compression="zstd",
    )
    class_counts = {
        str(int(key)): int(value)
        for key, value in labels[LABEL_COLUMN].value_counts().sort_index().items()
    }
    manifest = {
        "schema_version": "schedule-label-snapshot-v1",
        "build_id": build_id,
        "source_type": "snowflake_sql",
        "development_only": False,
        "source_sql_path": str(sql_path),
        "source_sql_sha256": sql_hash,
        "snowflake_query_ids": statement_ids,
        "target_definition_version": target_version,
        "created_at_utc": utc_now(),
        "row_count": int(len(labels)),
        "excluded_row_count": int(len(exclusions)),
        "class_counts": class_counts,
        "customers": sorted(labels["CUSTOMERNAME"].unique().tolist()),
        "labels_file": "labels.parquet",
    }
    write_json_atomic(staging / "manifest.json", manifest)
    write_json_atomic(staging / "profile.json", {
        "class_counts": class_counts,
        "exclusion_counts": exclusions[exclusion_column].fillna(
            "invalid_or_missing_label"
        ).value_counts().to_dict(),
        "percent_delayed_quantiles": {
            str(q): float(labels[TARGET_COLUMN].quantile(q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
    })
    write_checksums(staging, [
        "labels.parquet", "exclusions.parquet", "manifest.json", "profile.json"
    ])
    (staging / "COMPLETE").write_text("", encoding="ascii")
    staging.rename(final_dir)
    return final_dir
