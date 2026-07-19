from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import DataContractError, KEY_COLUMNS, LABEL_COLUMN, TARGET_COLUMN


PROHIBITED_EXACT = {
    "CUSTOMERNAME", "PROJECTID", "PROJECTNAME", "PROJECTCODE",
    "PROJECTDESCRIPTION", "PROJECTSTATUS", TARGET_COLUMN, LABEL_COLUMN,
}
PROHIBITED_PREFIXES = ("TARGET", "PAYMENT", "POSTING", "CHANGEORDER", "CO_")


def load_feature_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataContractError("Invalid candidate feature manifest: {}".format(exc)) from exc
    if not isinstance(manifest.get("features"), list):
        raise DataContractError("Candidate feature manifest requires a features list")
    return manifest


def generate_feature_manifest(
    schema_path: Path,
    output_path: Path,
    source_name: str = "approved_schedule_feature_schema",
) -> Dict[str, Any]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    features = []
    for name in schema["ordered_features"]:
        upper = str(name).upper()
        if upper.startswith("PROJ_KW_"):
            family = "project_keyword"
        elif upper.startswith("CONTRACT_KW_"):
            family = "contract_keyword"
        elif upper.startswith("ITEM_KW_"):
            family = "item_keyword"
        else:
            family = "approved_non_keyword"
        features.append({
            "name": upper,
            "family": family,
            "source": source_name,
            "expected_type": "numeric",
            "beginning_availability": "approved_or_pending_client_confirmation",
            "approved": True,
            "inference_schema_required": True,
        })
    manifest = {
        "schema_version": "schedule-candidate-features-v1",
        "feature_schema_version": schema["feature_schema_version"],
        "keyword_manifest_version": schema["keyword_manifest_version"],
        "features": features,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _vector_signature(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def qualify_features(
    development: pd.DataFrame,
    manifest: Dict[str, Any],
    maximum_missing_rate: float,
    minimum_non_null_count: int,
    drop_zero_variance: bool,
    drop_duplicate_vectors: bool,
) -> Tuple[List[str], pd.DataFrame]:
    records = []
    accepted = []
    signatures = {}
    for position, entry in enumerate(manifest["features"]):
        name = str(entry.get("name", "")).upper()
        reasons = []
        if not name:
            reasons.append("missing_feature_name")
        if name in PROHIBITED_EXACT or name.startswith(PROHIBITED_PREFIXES):
            reasons.append("prohibited_feature")
        if not bool(entry.get("approved", False)):
            reasons.append("not_approved")
        if not bool(entry.get("inference_schema_required", False)):
            reasons.append("not_in_inference_schema")
        if name not in development.columns:
            reasons.append("missing_from_snapshot")

        numeric = pd.Series(dtype=float)
        non_null = 0
        missing_rate = 1.0
        unique_count = 0
        finite_count = 0
        zero_variance = True
        minimum = maximum = mean = standard_deviation = None
        quantiles = {}
        duplicate_of = None

        if name in development.columns:
            numeric = pd.to_numeric(development[name], errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            non_null = int(numeric.notna().sum())
            missing_rate = float(numeric.isna().mean())
            unique_count = int(numeric.nunique(dropna=True))
            finite_count = int(np.isfinite(numeric.dropna()).sum())
            zero_variance = unique_count <= 1
            if non_null:
                minimum = float(numeric.min())
                maximum = float(numeric.max())
                mean = float(numeric.mean())
                standard_deviation = (
                    float(numeric.std(ddof=1)) if non_null > 1 else 0.0
                )
                quantiles = {
                    str(q): float(numeric.quantile(q))
                    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
                }
            if non_null < minimum_non_null_count:
                reasons.append("insufficient_non_null_count")
            if missing_rate > maximum_missing_rate:
                reasons.append("excessive_missing_rate")
            if drop_zero_variance and zero_variance:
                reasons.append("zero_variance")
            if not reasons and drop_duplicate_vectors:
                signature = _vector_signature(numeric)
                duplicate_of = signatures.get(signature)
                if duplicate_of:
                    reasons.append("duplicate_feature_vector")
                else:
                    signatures[signature] = name

        accepted_flag = not reasons
        if accepted_flag:
            accepted.append(name)
        records.append({
            "manifest_position": position,
            "feature_name": name,
            "family": entry.get("family"),
            "source": entry.get("source"),
            "accepted": accepted_flag,
            "rejection_reason": "|".join(reasons) if reasons else None,
            "non_null_count": non_null,
            "missing_rate": missing_rate,
            "unique_count": unique_count,
            "finite_count": finite_count,
            "minimum": minimum,
            "maximum": maximum,
            "mean": mean,
            "standard_deviation": standard_deviation,
            "quantiles_json": json.dumps(quantiles, sort_keys=True),
            "zero_variance": zero_variance,
            "duplicate_of": duplicate_of,
        })
    if not accepted:
        raise DataContractError("Feature qualification accepted no features")
    return accepted, pd.DataFrame(records)


def numeric_matrix(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    missing = [name for name in features if name not in frame]
    if missing:
        raise DataContractError("Matrix missing required features: " + ", ".join(missing[:20]))
    matrix = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan).astype(np.float32)
