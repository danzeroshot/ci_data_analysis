from pathlib import Path
import json
import re
import shutil
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import pandas as pd

from .contracts import CLASS_IDS, LABEL_COLUMN
from .lineage import compact_utc_now, environment_metadata, utc_now, write_checksums, write_json_atomic
from .metrics import classification_metrics
from .random_forest import build_pipeline, tune_random_forest
from .release import evaluate_release_gates, verify_bundle
from .splits import make_cv_assignments


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def _counts(labels: pd.Series) -> Dict[str, int]:
    counts = labels.value_counts()
    return {str(c): int(counts.get(c, 0)) for c in CLASS_IDS}


def _support(labels: pd.Series, minimum_rows: int, minimum_class: int) -> Dict[str, Any]:
    counts = _counts(labels)
    failed = []
    if len(labels) < minimum_rows:
        failed.append("minimum_rows")
    if any(counts[str(c)] < minimum_class for c in CLASS_IDS):
        failed.append("minimum_class_support")
    return {
        "rows": int(len(labels)),
        "class_counts": counts,
        "minimum_rows": minimum_rows,
        "minimum_rows_per_class": minimum_class,
        "eligible": not failed,
        "failed_criteria": failed,
    }


def _copy_parent_bundle(parent: Path, target: Path):
    shutil.copytree(parent, target)
    for name in ("checksums.sha256", "artifact_manifest.json"):
        path = target / name
        if path.exists():
            path.unlink()


def train_customer_models(
    joined: pd.DataFrame,
    development_mask: np.ndarray,
    matrix: pd.DataFrame,
    parameters: Dict[str, Any],
    config: Any,
    run_dir: Path,
    parent_candidate: Path,
    parent_model_version: str,
    qualified_schema: Dict[str, Any],
    feature_customers: Iterable[str],
    policy: Dict[str, Any],
    development_only_input: bool,
) -> Dict[str, Any]:
    output = run_dir / "customer_models"
    output.mkdir(exist_ok=True)
    dev = joined.loc[development_mask].reset_index(drop=True)
    hold = joined.loc[~development_mask].reset_index(drop=True)
    dev_matrix = matrix.loc[development_mask].reset_index(drop=True)
    hold_matrix = matrix.loc[~development_mask].reset_index(drop=True)
    customers = sorted(set(map(str, feature_customers)) | set(joined["CUSTOMERNAME"].astype(str)))
    entries = []
    for customer in customers:
        dev_mask = dev["CUSTOMERNAME"].astype(str).eq(customer).to_numpy()
        hold_mask = hold["CUSTOMERNAME"].astype(str).eq(customer).to_numpy()
        dev_labels_series = dev.loc[dev_mask, LABEL_COLUMN]
        hold_labels_series = hold.loc[hold_mask, LABEL_COLUMN]
        tuning_dev = _support(dev_labels_series, config.customer_models.tuning_development_minimum_rows, config.customer_models.tuning_development_minimum_rows_per_class)
        tuning_hold = _support(hold_labels_series, config.customer_models.tuning_holdout_minimum_rows, config.customer_models.tuning_holdout_minimum_rows_per_class)
        floor_dev = _support(dev_labels_series, config.customer_models.absolute_development_minimum_rows, config.customer_models.absolute_development_minimum_rows_per_class)
        floor_hold = _support(hold_labels_series, config.customer_models.absolute_holdout_minimum_rows, config.customer_models.absolute_holdout_minimum_rows_per_class)
        base = {"customer": customer, "tuning_development": tuning_dev, "tuning_holdout": tuning_hold, "absolute_development": floor_dev, "absolute_holdout": floor_hold}
        if not floor_dev["eligible"] or not floor_hold["eligible"]:
            entries.append(dict(base, status="unavailable", training_mode=None, reason_code="insufficient_customer_training_support"))
            continue
        customer_dev_matrix = dev_matrix.loc[dev_mask].reset_index(drop=True)
        customer_hold_matrix = hold_matrix.loc[hold_mask].reset_index(drop=True)
        customer_dev_labels = dev_labels_series.to_numpy(dtype=int)
        customer_hold_labels = hold_labels_series.to_numpy(dtype=int)
        if tuning_dev["eligible"] and tuning_hold["eligible"]:
            customer_dev = dev.loc[dev_mask].reset_index(drop=True)
            _, split_indexes = make_cv_assignments(customer_dev, config.tuning.cross_validation_folds, config.splits.split_seed)
            search = tune_random_forest(customer_dev_matrix, customer_dev_labels, split_indexes, config.tuning, config.selection.significant_delay_weight, config.random_seed, config.resources.model_n_jobs, config.reporting.calibration_bins)
            selected_parameters = search["selected_parameters"]
            selected_candidate = search["selected_candidate_id"]
            training_mode = "customer_tuned"
        else:
            selected_parameters = dict(parameters)
            selected_candidate = None
            training_mode = "global_parameters_customer_fit"
        pipeline = build_pipeline(selected_parameters, config.random_seed, config.resources.model_n_jobs)
        pipeline.fit(customer_dev_matrix, customer_dev_labels)
        hold_probability = pipeline.predict_proba(customer_hold_matrix)
        hold_prediction = pipeline.predict(customer_hold_matrix)
        hold_metrics = classification_metrics(customer_hold_labels, hold_prediction, hold_probability, config.selection.significant_delay_weight, config.reporting.calibration_bins)
        version = "schedule-rf-{}-{}-{}".format(compact_utc_now(), _safe_name(customer), parent_model_version[-8:])
        customer_dir = output / _safe_name(customer)
        candidate_dir = customer_dir / "candidate"
        customer_dir.mkdir(parents=True, exist_ok=True)
        _copy_parent_bundle(parent_candidate, candidate_dir)
        joblib.dump(pipeline, candidate_dir / "schedule_risk_model.joblib", compress=3)
        parity_input = pd.read_parquet(candidate_dir / "parity_input.parquet")
        parity_probability = pipeline.predict_proba(parity_input)
        parity_prediction = pipeline.predict(parity_input)
        write_json_atomic(candidate_dir / "parity_expected_predictions.json", {
            "predictions": [int(value) for value in parity_prediction],
            "probabilities": parity_probability.tolist(),
            "absolute_tolerance": 1e-12,
        })
        gates = evaluate_release_gates(hold_metrics, policy, development_only_input, True)
        status = "rejected" if any(g["status"] == "fail" for g in gates) else ("development_candidate" if development_only_input else "host_validated")
        write_json_atomic(candidate_dir / "selected_hyperparameters.json", {"candidate_id": selected_candidate, "parameters": selected_parameters, "significant_delay_weight": config.selection.significant_delay_weight, "training_mode": training_mode, "parent_all_customer_model_version": parent_model_version})
        write_json_atomic(candidate_dir / "schedule_risk_training_metrics.json", {"schema_version": "schedule-customer-training-metrics-v1", "model_version": version, "parent_all_customer_model_version": parent_model_version, "customer": customer, "training_mode": training_mode, "development": floor_dev, "locked_holdout": hold_metrics, "release_gates": gates})
        write_json_atomic(candidate_dir / "schedule_risk_model_card.json", {"schema_version": "schedule-customer-model-card-v1", "model_version": version, "parent_all_customer_model_version": parent_model_version, "customer": customer, "status": status, "created_at_utc": utc_now(), "target_definition_version": config.target_definition_version, "target": "PercentDelayed bins <=0, (0,25], >25", "feature_schema_version": qualified_schema["feature_schema_version"], "keyword_manifest_version": qualified_schema["keyword_manifest_version"], "feature_count": len(qualified_schema["ordered_features"]), "training_rows": int(len(customer_dev_labels)), "locked_holdout_rows": int(len(customer_hold_labels)), "selected_hyperparameters": selected_parameters, "training_mode": training_mode, "development_only_input": development_only_input, "headline_locked_holdout_metrics": hold_metrics["overall"], "release_gates": gates, "training_environment": environment_metadata()})
        files = sorted(path.name for path in candidate_dir.iterdir() if path.is_file() and path.name not in {"checksums.sha256", "artifact_manifest.json"})
        write_json_atomic(candidate_dir / "artifact_manifest.json", {"schema_version": "schedule-customer-artifact-manifest-v1", "model_version": version, "parent_all_customer_model_version": parent_model_version, "customer": customer, "files": files})
        write_checksums(candidate_dir, files + ["artifact_manifest.json"])
        verify_bundle(candidate_dir)
        entries.append(dict(base, status=status, training_mode=training_mode, model_version=version, parent_all_customer_model_version=parent_model_version, performance_gates=gates, holdout_metrics=hold_metrics, candidate_dir=str(candidate_dir)))
    result = {"schema_version": "schedule-customer-models-v1", "parent_all_customer_model_version": parent_model_version, "customers": entries}
    write_json_atomic(output / "customer_model_eligibility.json", result)
    pd.DataFrame([{"customer": e["customer"], "status": e["status"], "training_mode": e.get("training_mode"), "reason_code": e.get("reason_code"), "development_rows": e["absolute_development"]["rows"], "holdout_rows": e["absolute_holdout"]["rows"], "development_class_counts_json": json.dumps(e["absolute_development"]["class_counts"]), "holdout_class_counts_json": json.dumps(e["absolute_holdout"]["class_counts"])} for e in entries]).to_csv(output / "customer_model_eligibility.csv", index=False)
    return result
