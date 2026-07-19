from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from .contracts import ReleaseGateError
from .lineage import sha256_file, utc_now, verify_checksums, write_json_atomic


REQUIRED_BUNDLE_FILES = (
    "schedule_risk_model.joblib",
    "schedule_risk_feature_schema.json",
    "schedule_risk_model_card.json",
    "schedule_risk_training_metrics.json",
    "selected_hyperparameters.json",
    "reference_metrics.json",
    "feature_reference_profile.parquet",
    "prediction_reference_profile.json",
    "histogram_definitions.json",
    "parity_input.parquet",
    "parity_expected_predictions.json",
    "requirements.lock",
    "artifact_manifest.json",
    "checksums.sha256",
)


def verify_bundle(bundle_dir: Path) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle_dir / name).is_file()]
    if missing:
        raise ReleaseGateError("Bundle missing required files: " + ", ".join(missing))
    verify_checksums(bundle_dir)
    schema = json.loads((bundle_dir / "schedule_risk_feature_schema.json").read_text())
    card = json.loads((bundle_dir / "schedule_risk_model_card.json").read_text())
    pipeline = joblib.load(bundle_dir / "schedule_risk_model.joblib")
    classifier = pipeline.named_steps["classifier"]
    if [int(value) for value in classifier.classes_] != [0, 1, 2]:
        raise ReleaseGateError("Serialized model classes must be [0, 1, 2]")
    if not schema.get("ordered_features"):
        raise ReleaseGateError("Released feature schema is empty")
    if not card.get("model_version"):
        raise ReleaseGateError("Model card has no model version")
    return {"schema": schema, "model_card": card}


def evaluate_release_gates(
    holdout_metrics: Dict[str, Any],
    policy: Dict[str, Any],
    development_only_input: bool,
    parity_passed: bool,
    subgroup_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    overall = holdout_metrics["overall"]
    gates = [
        {
            "name": "serialization_parity",
            "status": "pass" if parity_passed else "fail",
            "detail": "Serialized predictions match pre-serialization predictions",
        },
        {
            "name": "production_input_lineage",
            "status": "warn" if development_only_input else "pass",
            "detail": (
                "Legacy CSV labels restrict this candidate to development status"
                if development_only_input else "Label snapshot is production eligible"
            ),
        },
    ]
    if subgroup_result is not None:
        required = policy.get(
            "required_subgroup_families",
            [
                "planned_duration",
                "planned_value",
                "contract_item_count",
                "predictor_missingness",
            ],
        )
        failed = [
            family for family in required
            if not subgroup_result.get("families", {}).get(family, {}).get("eligible", False)
        ]
        gates.append({
            "name": "required_subgroup_family_eligibility",
            "status": "pass" if not failed else "fail",
            "detail": {
                "required_families": required,
                "failed_families": failed,
            },
        })
    thresholds = policy.get("metric_thresholds", {})
    metric_map = {
        "minimum_macro_f1": ("macro_f1", "minimum"),
        "minimum_significant_delay_recall": ("significant_delay_recall", "minimum"),
        "minimum_balanced_accuracy": ("balanced_accuracy", "minimum"),
        "maximum_calibration_error": ("expected_calibration_error", "maximum"),
    }
    for policy_name, (metric_name, direction) in metric_map.items():
        threshold = thresholds.get(policy_name)
        if threshold is None:
            gates.append({
                "name": policy_name,
                "status": "warn",
                "detail": "Threshold is not approved",
            })
            continue
        value = overall.get(metric_name)
        passed = value is not None and (
            value >= threshold if direction == "minimum" else value <= threshold
        )
        gates.append({
            "name": policy_name,
            "status": "pass" if passed else "fail",
            "detail": "{}={} threshold={}".format(metric_name, value, threshold),
        })
    return gates


def promote_run(
    run_dir: Path,
    output_root: Path,
    production: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    candidate = run_dir / "candidate"
    verification = verify_bundle(candidate)
    result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    gates = result.get("release_gates", [])
    if any(gate["status"] == "fail" for gate in gates):
        raise ReleaseGateError("Candidate has failed release gates")
    if production:
        if result.get("development_only_input"):
            raise ReleaseGateError("Development-only labels cannot be promoted to production")
        if any(gate["status"] != "pass" for gate in gates):
            raise ReleaseGateError("Production promotion requires every gate to pass")
    model_version = verification["model_card"]["model_version"]
    releases = Path(output_root) / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    final = releases / model_version
    if final.exists():
        raise ReleaseGateError("Release already exists: {}".format(model_version))
    staging = releases / ("." + model_version + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(candidate, staging)
    verify_bundle(staging)
    staging.rename(final)
    pointer = {
        "schema_version": "schedule-release-pointer-v1",
        "model_version": model_version,
        "relative_path": "releases/{}".format(model_version),
        "artifact_manifest_sha256": sha256_file(final / "artifact_manifest.json"),
        "promoted_at_utc": utc_now(),
        "promotion_status": "production_approved" if production else "host_validated",
    }
    write_json_atomic(Path(output_root) / "current.json", pointer)
    return final


def rollback_release(output_root: Path, model_version: str) -> Path:
    output_root = Path(output_root)
    release = output_root / "releases" / model_version
    verification = verify_bundle(release)
    pointer = {
        "schema_version": "schedule-release-pointer-v1",
        "model_version": model_version,
        "relative_path": "releases/{}".format(model_version),
        "artifact_manifest_sha256": sha256_file(release / "artifact_manifest.json"),
        "promoted_at_utc": utc_now(),
        "promotion_status": verification["model_card"].get("status", "host_validated"),
        "rollback": True,
    }
    write_json_atomic(output_root / "current.json", pointer)
    return release


def resolve_current_release(output_root: Path) -> Path:
    pointer = json.loads((Path(output_root) / "current.json").read_text(encoding="utf-8"))
    relative = Path(pointer["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseGateError("Unsafe release pointer path")
    release = (Path(output_root) / relative).resolve()
    verify_bundle(release)
    return release


def promote_customer_bundle(
    candidate_dir: Path,
    output_root: Path,
    customer: str,
    production: bool = False,
) -> Path:
    candidate_dir = Path(candidate_dir)
    verification = verify_bundle(candidate_dir)
    card = verification["model_card"]
    gates = card.get("release_gates", [])
    if any(gate.get("status") == "fail" for gate in gates):
        raise ReleaseGateError("Customer candidate has failed release gates")
    if production:
        if card.get("development_only_input"):
            raise ReleaseGateError("Development-only labels cannot be promoted")
        if any(gate.get("status") != "pass" for gate in gates):
            raise ReleaseGateError("Production promotion requires every gate to pass")
    customer_dir = Path(output_root) / "customer-releases" / _safe_release_name(customer)
    releases = customer_dir / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    model_version = verification["model_card"]["model_version"]
    final = releases / model_version
    if final.exists():
        raise ReleaseGateError("Customer release already exists: " + model_version)
    staging = releases / ("." + model_version + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(candidate_dir, staging)
    verify_bundle(staging)
    staging.rename(final)
    pointer = {
        "schema_version": "schedule-customer-release-pointer-v1",
        "customer": customer,
        "model_version": model_version,
        "relative_path": "releases/{}".format(model_version),
        "artifact_manifest_sha256": sha256_file(final / "artifact_manifest.json"),
        "promoted_at_utc": utc_now(),
        "promotion_status": "production_approved" if production else "host_validated",
    }
    write_json_atomic(customer_dir / "current.json", pointer)
    return final


def _safe_release_name(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"
