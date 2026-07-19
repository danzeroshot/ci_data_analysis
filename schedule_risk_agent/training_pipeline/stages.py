from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .configuration import TrainingRunConfig, load_run_config
from .contracts import CLASS_LABELS, LABEL_COLUMN
from .feature_qualification import (
    load_feature_manifest,
    numeric_matrix,
    qualify_features,
)
from .lineage import (
    compact_utc_now,
    environment_metadata,
    make_run_id,
    sha256_file,
    sha256_json,
    stage_status,
    utc_now,
    write_checksums,
    write_json_atomic,
)
from .metrics import bootstrap_confidence_intervals, classification_metrics
from .profiles import (
    feature_reference_profile,
    population_stability_index,
    prediction_reference_profile,
)
from .random_forest import build_pipeline, tune_random_forest
from .subgroups import evaluate_subgroups
from .customer_models import train_customer_models
from .release import evaluate_release_gates, verify_bundle
from .reporting import generate_report
from .snapshots import (
    join_training_data,
    load_feature_snapshot,
    load_label_snapshot,
)
from .splits import (
    assign_locked_holdout,
    customer_holdout_splits,
    make_cv_assignments,
    temporal_split,
)


def _json_safe_search(search: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in search.items() if key != "oof_predictions"
    }


def _candidate_summary(search: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    ranks = {
        candidate_id: rank + 1
        for rank, candidate_id in enumerate(search["ranked_candidate_ids"])
    }
    for item in search["candidate_results"]:
        aggregate = item.get("aggregate") or {}
        overall = aggregate.get("overall") or {}
        rows.append({
            "candidate_id": item["candidate_id"],
            "status": item["status"],
            "rank": ranks.get(item["candidate_id"]),
            "elapsed_seconds": item["elapsed_seconds"],
            "failure": item.get("failure"),
            "selection_score": overall.get("selection_score"),
            "macro_f1": overall.get("macro_f1"),
            "balanced_accuracy": overall.get("balanced_accuracy"),
            "significant_delay_recall": overall.get("significant_delay_recall"),
            "train_to_validation_macro_f1_gap": aggregate.get(
                "train_to_validation_macro_f1_gap"
            ),
            "parameters_json": json.dumps(item["parameters"], sort_keys=True),
        })
    return pd.DataFrame(rows)


def _metrics_long(
    contexts: List[Tuple[str, Dict[str, Any]]],
    run_id: str,
    model_version: str,
    config: TrainingRunConfig,
) -> pd.DataFrame:
    rows = []
    for population, metrics in contexts:
        for name, value in metrics.get("overall", {}).items():
            rows.append({
                "metric_schema_version": "schedule-metrics-v1",
                "run_id": run_id,
                "model_version": model_version,
                "target_definition_version": config.target_definition_version,
                "evaluation_population": population,
                "class_label": None,
                "metric_name": name,
                "value": value,
                "undefined_reason": "not_defined" if value is None else None,
                "random_seed": config.random_seed,
                "evaluated_at_utc": utc_now(),
            })
        for class_label, class_metrics in metrics.get("per_class", {}).items():
            for name, value in class_metrics.items():
                if name == "class_id" or name.endswith("undefined_reason"):
                    continue
                rows.append({
                    "metric_schema_version": "schedule-metrics-v1",
                    "run_id": run_id,
                    "model_version": model_version,
                    "target_definition_version": config.target_definition_version,
                    "evaluation_population": population,
                    "class_label": class_label,
                    "metric_name": name,
                    "value": value,
                    "undefined_reason": (
                        class_metrics.get(name + "_undefined_reason")
                        if value is None else None
                    ),
                    "random_seed": config.random_seed,
                    "evaluated_at_utc": utc_now(),
                })
    return pd.DataFrame(rows)


def _evaluate_stress_tests(
    joined: pd.DataFrame,
    matrix: pd.DataFrame,
    parameters: Dict[str, Any],
    config: TrainingRunConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    labels = joined[LABEL_COLUMN].to_numpy(dtype=int)
    temporal_result = {"available": False, "reason": "disabled"}
    if config.splits.run_temporal_test:
        plan = temporal_split(
            joined,
            config.splits.temporal_date_column,
            config.splits.temporal_test_fraction,
        )
        temporal_result = {
            key: value for key, value in plan.items()
            if key not in {"train_index", "test_index"}
        }
        if plan.get("available"):
            pipeline = build_pipeline(
                parameters, config.random_seed, config.resources.model_n_jobs
            )
            pipeline.fit(matrix.iloc[plan["train_index"]], labels[plan["train_index"]])
            probability = pipeline.predict_proba(matrix.iloc[plan["test_index"]])
            prediction = pipeline.predict(matrix.iloc[plan["test_index"]])
            temporal_result["metrics"] = classification_metrics(
                labels[plan["test_index"]],
                prediction,
                probability,
                config.selection.significant_delay_weight,
                config.reporting.calibration_bins,
            )

    customer_result = {"enabled": config.splits.run_customer_tests, "customers": []}
    if config.splits.run_customer_tests:
        for plan in customer_holdout_splits(
            joined,
            config.splits.minimum_customer_rows,
            config.splits.minimum_customer_class_support,
        ):
            stored = {
                key: value for key, value in plan.items()
                if key not in {"train_index", "test_index"}
            }
            if plan["eligible"]:
                pipeline = build_pipeline(
                    parameters, config.random_seed, config.resources.model_n_jobs
                )
                pipeline.fit(matrix.iloc[plan["train_index"]], labels[plan["train_index"]])
                probability = pipeline.predict_proba(matrix.iloc[plan["test_index"]])
                prediction = pipeline.predict(matrix.iloc[plan["test_index"]])
                stored["metrics"] = classification_metrics(
                    labels[plan["test_index"]],
                    prediction,
                    probability,
                    config.selection.significant_delay_weight,
                    config.reporting.calibration_bins,
                )
            customer_result["customers"].append(stored)
    return temporal_result, customer_result


def _release_policy(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_pipeline(config_path: Path, stop_after: Optional[str] = None) -> Path:
    config = load_run_config(config_path)
    normalized = config.model_dump(mode="json")
    config_hash = sha256_json(normalized)
    run_id = make_run_id(config.run_name, config_hash)
    run_dir = config.output_root / "runs" / run_id
    if run_dir.exists():
        raise RuntimeError("Run directory already exists: {}".format(run_dir))
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "run_config.json", normalized)
    write_json_atomic(run_dir / "environment.json", environment_metadata())
    write_json_atomic(run_dir / "input_manifest.json", {
        "schema_version": "schedule-training-input-manifest-v1",
        "config_sha256": config_hash,
        "feature_data_sha256": sha256_file(config.training_snapshot.path),
        "feature_manifest_sha256": sha256_file(config.training_snapshot.manifest_path),
        "label_data_sha256": sha256_file(config.labels.path),
        "label_manifest_sha256": sha256_file(config.labels.manifest_path),
        "candidate_feature_manifest_sha256": sha256_file(
            config.feature_policy.manifest_path
        ),
        "release_policy_sha256": sha256_file(config.release_policy_path),
    })

    with stage_status(run_dir, "inputs"):
        features, feature_snapshot_manifest = load_feature_snapshot(
            config.training_snapshot.path,
            config.training_snapshot.manifest_path,
        )
        labels, label_snapshot_manifest = load_label_snapshot(
            config.labels.path,
            config.labels.manifest_path,
        )
        joined, reconciliation = join_training_data(features, labels)
        write_json_atomic(run_dir / "reconciliation.json", reconciliation)
        write_json_atomic(run_dir / "feature_snapshot_manifest.json", feature_snapshot_manifest)
        write_json_atomic(run_dir / "label_snapshot_manifest.json", label_snapshot_manifest)

    with stage_status(run_dir, "splits"):
        assignments = assign_locked_holdout(
            joined,
            config.target_definition_version,
            config.splits.locked_holdout_fraction,
            config.splits.split_seed,
            config.splits.minimum_class_support,
        )
        development_mask = assignments["POPULATION"].eq("development").to_numpy()
        holdout_mask = assignments["POPULATION"].eq("locked_holdout").to_numpy()
        development = joined.loc[development_mask].reset_index(drop=True)
        holdout = joined.loc[holdout_mask].reset_index(drop=True)
        cv_assignments, split_indexes = make_cv_assignments(
            development,
            config.tuning.cross_validation_folds,
            config.splits.split_seed,
        )
        assignments.to_parquet(run_dir / "split_assignments.parquet", index=False)
        cv_assignments.to_parquet(run_dir / "cv_split_assignments.parquet", index=False)
        write_json_atomic(run_dir / "split_summary.json", {
            "development_rows": int(development_mask.sum()),
            "locked_holdout_rows": int(holdout_mask.sum()),
            "development_class_counts": development[LABEL_COLUMN].value_counts().sort_index().to_dict(),
            "holdout_class_counts": holdout[LABEL_COLUMN].value_counts().sort_index().to_dict(),
            "cv_folds": int(cv_assignments["CVFOLD"].nunique()),
        })

    with stage_status(run_dir, "qualification"):
        candidate_manifest = load_feature_manifest(config.feature_policy.manifest_path)
        accepted, qualification = qualify_features(
            development,
            candidate_manifest,
            config.feature_policy.maximum_missing_rate,
            config.feature_policy.minimum_non_null_count,
            config.feature_policy.drop_zero_variance,
            config.feature_policy.drop_duplicate_vectors,
        )
        qualification.to_parquet(run_dir / "feature_qualification.parquet", index=False)
        qualification.to_csv(run_dir / "feature_qualification.csv", index=False)
        qualified_schema = {
            "feature_schema_version": feature_snapshot_manifest.get(
                "feature_schema_version", candidate_manifest.get("feature_schema_version")
            ),
            "keyword_manifest_version": feature_snapshot_manifest.get(
                "keyword_manifest_version", candidate_manifest.get("keyword_manifest_version")
            ),
            "ordered_features": accepted,
            "class_labels": list(CLASS_LABELS),
        }
        write_json_atomic(run_dir / "qualified_feature_schema.json", qualified_schema)
        write_json_atomic(run_dir / "qualification_summary.json", {
            "candidate_count": int(len(qualification)),
            "accepted_count": int(qualification["accepted"].sum()),
            "rejected_count": int((~qualification["accepted"]).sum()),
            "rejection_counts": qualification["rejection_reason"].value_counts().to_dict(),
        })
    if stop_after == "qualify":
        write_json_atomic(run_dir / "run_result.json", {
            "run_id": run_id, "status": "qualified", "run_dir": str(run_dir)
        })
        return run_dir

    with stage_status(run_dir, "tuning"):
        all_matrix = numeric_matrix(joined, accepted)
        development_matrix = all_matrix.loc[development_mask].reset_index(drop=True)
        holdout_matrix = all_matrix.loc[holdout_mask].reset_index(drop=True)
        development_labels = development[LABEL_COLUMN].to_numpy(dtype=int)
        search = tune_random_forest(
            development_matrix,
            development_labels,
            split_indexes,
            config.tuning,
            config.selection.significant_delay_weight,
            config.random_seed,
            config.resources.model_n_jobs,
            config.reporting.calibration_bins,
        )
        write_json_atomic(
            run_dir / "hyperparameter_search_results.json",
            _json_safe_search(search),
        )
        summary = _candidate_summary(search)
        summary.to_parquet(run_dir / "hyperparameter_search_results.parquet", index=False)
        summary.to_csv(run_dir / "hyperparameter_search_results.csv", index=False)
        selected_oof = search["oof_predictions"].loc[
            search["oof_predictions"]["candidate_id"].eq(search["selected_candidate_id"])
        ].copy()
        selected_oof.to_parquet(run_dir / "cv_out_of_fold_predictions.parquet", index=False)
        write_json_atomic(run_dir / "selected_hyperparameters.json", {
            "candidate_id": search["selected_candidate_id"],
            "parameters": search["selected_parameters"],
            "significant_delay_weight": config.selection.significant_delay_weight,
        })
    if stop_after == "tune":
        write_json_atomic(run_dir / "run_result.json", {
            "run_id": run_id, "status": "tuned", "run_dir": str(run_dir)
        })
        return run_dir

    with stage_status(run_dir, "final_evaluation"):
        parameters = search["selected_parameters"]
        final_pipeline = build_pipeline(
            parameters, config.random_seed, config.resources.model_n_jobs
        )
        fit_started = time.time()
        final_pipeline.fit(development_matrix, development_labels)
        fit_seconds = time.time() - fit_started
        train_probability = final_pipeline.predict_proba(development_matrix)
        train_prediction = final_pipeline.predict(development_matrix)
        holdout_probability = final_pipeline.predict_proba(holdout_matrix)
        holdout_prediction = final_pipeline.predict(holdout_matrix)
        holdout_labels = holdout[LABEL_COLUMN].to_numpy(dtype=int)
        train_metrics = classification_metrics(
            development_labels, train_prediction, train_probability,
            config.selection.significant_delay_weight,
            config.reporting.calibration_bins,
        )
        holdout_metrics = classification_metrics(
            holdout_labels, holdout_prediction, holdout_probability,
            config.selection.significant_delay_weight,
            config.reporting.calibration_bins,
        )
        holdout_metrics["fit_seconds"] = fit_seconds
        confidence_intervals = bootstrap_confidence_intervals(
            holdout_labels, holdout_prediction, holdout_probability,
            config.selection.significant_delay_weight,
            config.resources.bootstrap_iterations,
            config.random_seed,
        )
        holdout_predictions = holdout[["CUSTOMERNAME", "PROJECTID", LABEL_COLUMN]].copy()
        holdout_predictions["PREDICTED"] = holdout_prediction
        for class_id, label in enumerate(CLASS_LABELS):
            holdout_predictions["PROBABILITY_" + label.upper()] = holdout_probability[:, class_id]
        holdout_predictions.to_parquet(
            run_dir / "locked_holdout_predictions.parquet", index=False
        )
        temporal_metrics, customer_metrics = _evaluate_stress_tests(
            joined.reset_index(drop=True),
            all_matrix.reset_index(drop=True),
            parameters,
            config,
        )
        transformed_names = final_pipeline.named_steps["imputer"].get_feature_names_out(
            accepted
        )
        importance_values = final_pipeline.named_steps["classifier"].feature_importances_
        family_by_name = {
            str(entry.get("name", "")).upper(): entry.get("family", "unknown")
            for entry in candidate_manifest["features"]
        }
        importance_rows = []
        for name, importance in zip(transformed_names, importance_values):
            feature_name = str(name)
            source_name = feature_name.replace("missingindicator_", "", 1)
            importance_rows.append({
                "feature_name": feature_name,
                "source_feature_name": source_name,
                "family": (
                    "missingness_indicator"
                    if feature_name.startswith("missingindicator_")
                    else family_by_name.get(source_name, "unknown")
                ),
                "importance": float(importance),
            })
        feature_importance = pd.DataFrame(importance_rows).sort_values(
            ["importance", "feature_name"], ascending=[False, True]
        ).reset_index(drop=True)
        feature_importance.insert(0, "rank", np.arange(1, len(feature_importance) + 1))
        feature_importance.to_parquet(run_dir / "feature_importances.parquet", index=False)
        feature_importance.to_csv(run_dir / "feature_importances.csv", index=False)
        write_json_atomic(run_dir / "train_metrics.json", train_metrics)
        write_json_atomic(run_dir / "locked_holdout_metrics.json", holdout_metrics)
        write_json_atomic(run_dir / "confidence_intervals.json", confidence_intervals)
        write_json_atomic(run_dir / "temporal_metrics.json", temporal_metrics)
        write_json_atomic(run_dir / "customer_holdout_metrics.json", customer_metrics)

    subgroup_result = None
    if config.subgroup_evaluation.enabled:
        subgroup_oof = selected_oof.rename(columns={
            "actual": LABEL_COLUMN,
            "predicted": "PREDICTED",
            "probability_no_delay": "PROBABILITY_NO_DELAY",
            "probability_mild_delay": "PROBABILITY_MILD_DELAY",
            "probability_significant_delay": "PROBABILITY_SIGNIFICANT_DELAY",
        }).copy()
        development_keys = development[["CUSTOMERNAME", "PROJECTID"]].reset_index(drop=True)
        subgroup_oof["CUSTOMERNAME"] = development_keys.loc[subgroup_oof["row_index"], "CUSTOMERNAME"].to_numpy()
        subgroup_oof["PROJECTID"] = development_keys.loc[subgroup_oof["row_index"], "PROJECTID"].to_numpy()
        subgroup_result = evaluate_subgroups(
            joined.reset_index(drop=True),
            development_mask,
            accepted,
            subgroup_oof,
            holdout_predictions,
            config.subgroup_evaluation,
            config.selection.significant_delay_weight,
            config.reporting.calibration_bins,
        )
        subgroup_dir = run_dir / "subgroups"
        subgroup_dir.mkdir()
        assignments_frame = subgroup_result.pop("assignments")
        assignments_frame.to_parquet(subgroup_dir / "subgroup_assignments.parquet", index=False)
        assignments_frame.to_csv(subgroup_dir / "subgroup_assignments.csv", index=False)
        write_json_atomic(subgroup_dir / "subgroup_eligibility.json", {
            "schema_version": subgroup_result["schema_version"],
            "families": {
                name: {
                    "eligible": item["eligible"],
                    "bands": item["bands"],
                    "checks": item["checks"],
                    "support": item["support"],
                    "source_details": item["source_details"],
                }
                for name, item in subgroup_result["families"].items()
            },
        })
        metric_rows = []
        for family, item in subgroup_result["families"].items():
            for record in item["metrics"]:
                for metric_name, metric_value in record["metrics"].get("overall", {}).items():
                    metric_rows.append({
                        "family": family,
                        "population": record["population"],
                        "band": record["band"],
                        "class_label": None,
                        "metric_name": metric_name,
                        "value": metric_value,
                    })
        pd.DataFrame(metric_rows).to_parquet(
            subgroup_dir / "subgroup_metrics_long.parquet", index=False
        )
        pd.DataFrame(metric_rows).to_csv(
            subgroup_dir / "subgroup_metrics_long.csv", index=False
        )

    customer_model_result = None
    with stage_status(run_dir, "reference_profiles"):
        feature_profile, histograms = feature_reference_profile(development_matrix)
        feature_profile.to_parquet(run_dir / "feature_reference_profile.parquet", index=False)
        write_json_atomic(run_dir / "histogram_definitions.json", histograms)
        prediction_profile = prediction_reference_profile(
            holdout_labels, holdout_prediction, holdout_probability
        )
        write_json_atomic(run_dir / "prediction_reference_profile.json", prediction_profile)
        selected_candidate = next(
            item for item in search["candidate_results"]
            if item["candidate_id"] == search["selected_candidate_id"]
        )
        reference_metrics = {
            "schema_version": "schedule-reference-metrics-v1",
            "development_in_sample": train_metrics,
            "cross_validation_out_of_fold": selected_candidate["aggregate"],
            "locked_holdout": holdout_metrics,
            "confidence_intervals": confidence_intervals,
            "temporal": temporal_metrics,
            "customer_holdouts": customer_metrics,
            "subgroups": subgroup_result,
            "customer_models": customer_model_result,
        }
        write_json_atomic(run_dir / "reference_metrics.json", reference_metrics)

    with stage_status(run_dir, "bundle"):
        model_version = "schedule-rf-{}-{}".format(compact_utc_now(), config_hash[:8])
        candidate_dir = run_dir / "candidate"
        candidate_dir.mkdir()
        joblib.dump(final_pipeline, candidate_dir / "schedule_risk_model.joblib", compress=3)
        write_json_atomic(candidate_dir / "schedule_risk_feature_schema.json", qualified_schema)
        shutil.copy2(run_dir / "selected_hyperparameters.json", candidate_dir)
        shutil.copy2(run_dir / "feature_reference_profile.parquet", candidate_dir)
        shutil.copy2(run_dir / "prediction_reference_profile.json", candidate_dir)
        shutil.copy2(run_dir / "histogram_definitions.json", candidate_dir)
        shutil.copy2(run_dir / "reference_metrics.json", candidate_dir)
        shutil.copy2(run_dir / "feature_importances.parquet", candidate_dir)
        shutil.copy2(Path("requirements.lock"), candidate_dir / "requirements.lock")

        parity_count = min(20, len(holdout_matrix))
        parity_input = holdout_matrix.iloc[:parity_count].copy()
        parity_expected = {
            "predictions": [int(value) for value in holdout_prediction[:parity_count]],
            "probabilities": holdout_probability[:parity_count].tolist(),
            "absolute_tolerance": 1e-12,
        }
        parity_input.to_parquet(candidate_dir / "parity_input.parquet", index=False)
        write_json_atomic(
            candidate_dir / "parity_expected_predictions.json", parity_expected
        )
        reloaded = joblib.load(candidate_dir / "schedule_risk_model.joblib")
        reloaded_probability = reloaded.predict_proba(parity_input)
        reloaded_prediction = reloaded.predict(parity_input)
        parity_passed = bool(
            np.array_equal(reloaded_prediction, holdout_prediction[:parity_count])
            and np.allclose(
                reloaded_probability,
                holdout_probability[:parity_count],
                rtol=0.0,
                atol=parity_expected["absolute_tolerance"],
            )
        )
        policy = _release_policy(config.release_policy_path)
        development_only_input = bool(label_snapshot_manifest.get("development_only", True))
        release_gates = evaluate_release_gates(
            holdout_metrics, policy, development_only_input, parity_passed,
            subgroup_result,
        )
        status = "rejected" if any(
            gate["status"] == "fail" for gate in release_gates
        ) else "development_candidate" if development_only_input else "host_validated"
        metrics_bundle = {
            "schema_version": "schedule-training-metrics-v1",
            "model_version": model_version,
            "selection_weight": config.selection.significant_delay_weight,
            "selected_candidate_id": search["selected_candidate_id"],
            "train": train_metrics,
            "locked_holdout": holdout_metrics,
            "confidence_intervals": confidence_intervals,
            "release_gates": release_gates,
            "subgroup_eligibility": (
                None if subgroup_result is None else {
                    family: value["eligible"]
                    for family, value in subgroup_result["families"].items()
                }
            ),
        }
        write_json_atomic(
            candidate_dir / "schedule_risk_training_metrics.json", metrics_bundle
        )
        model_card = {
            "schema_version": "schedule-model-card-v1",
            "model_version": model_version,
            "run_id": run_id,
            "status": status,
            "created_at_utc": utc_now(),
            "target_definition_version": config.target_definition_version,
            "target": "PercentDelayed bins <=0, (0,25], >25",
            "feature_schema_version": qualified_schema["feature_schema_version"],
            "keyword_manifest_version": qualified_schema["keyword_manifest_version"],
            "feature_count": len(accepted),
            "training_rows": len(development),
            "locked_holdout_rows": len(holdout),
            "significant_delay_weight": config.selection.significant_delay_weight,
            "selected_hyperparameters": parameters,
            "headline_locked_holdout_metrics": holdout_metrics["overall"],
            "development_only_input": development_only_input,
            "training_environment": environment_metadata(),
            "limitations": [
                "Client approval of beginning-available fields remains pending",
                "Retrospective fields may require as-of-history validation",
                "Docker and Python 3.11 validation remain pending",
            ],
        }
        write_json_atomic(candidate_dir / "schedule_risk_model_card.json", model_card)
        artifact_manifest = {
            "schema_version": "schedule-model-artifact-manifest-v1",
            "model_version": model_version,
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "config_sha256": config_hash,
            "files": sorted([
                path.name for path in candidate_dir.iterdir()
                if path.is_file() and path.name not in {"checksums.sha256", "artifact_manifest.json"}
            ]),
        }
        write_json_atomic(candidate_dir / "artifact_manifest.json", artifact_manifest)
        checksum_names = [
            path.name for path in candidate_dir.iterdir()
            if path.is_file() and path.name != "checksums.sha256"
        ]
        write_checksums(candidate_dir, checksum_names)
        verify_bundle(candidate_dir)

    customer_model_result = None
    if config.customer_models.enabled:
        customer_model_result = train_customer_models(
            joined.reset_index(drop=True),
            development_mask,
            all_matrix.reset_index(drop=True),
            parameters,
            config,
            run_dir,
            candidate_dir,
            model_version,
            qualified_schema,
            feature_snapshot_manifest.get("customers", []),
            policy,
            development_only_input,
        )

    contexts = [
        ("development_in_sample", train_metrics),
        ("cross_validation_out_of_fold", selected_candidate["aggregate"]),
        ("locked_hash_holdout", holdout_metrics),
    ]
    if temporal_metrics.get("available") and temporal_metrics.get("metrics"):
        contexts.append(("temporal_holdout", temporal_metrics["metrics"]))
    for customer_result in customer_metrics.get("customers", []):
        if customer_result.get("metrics"):
            contexts.append((
                "customer_holdout:{}".format(customer_result["customer"]),
                customer_result["metrics"],
            ))
    metrics_long = _metrics_long(contexts, run_id, model_version, config)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir()
    metrics_long.to_parquet(metrics_dir / "metrics_long.parquet", index=False)
    metrics_long.to_csv(metrics_dir / "metrics_long.csv", index=False)

    customer_entries = customer_metrics.get("customers", [])
    labeled_customer_names = {
        item["customer"] for item in customer_entries
    }
    feature_customer_names = set(feature_snapshot_manifest.get("customers", []))
    if not feature_customer_names:
        feature_customer_names = labeled_customer_names
    missing_label_customers = sorted(feature_customer_names - labeled_customer_names)
    eligible_customer_count = sum(
        item.get("eligible", False) for item in customer_entries
    )
    if not config.splits.run_customer_tests:
        customer_evaluation_status = "not_run"
    elif not customer_entries or eligible_customer_count == 0:
        customer_evaluation_status = "unavailable"
    elif eligible_customer_count < len(customer_entries):
        customer_evaluation_status = "limited"
    else:
        customer_evaluation_status = "completed"

    run_result = {
        "schema_version": "schedule-training-run-result-v1",
        "run_id": run_id,
        "model_version": model_version,
        "status": status,
        "run_dir": str(run_dir),
        "candidate_dir": str(candidate_dir),
        "development_only_input": development_only_input,
        "accepted_feature_count": len(accepted),
        "selected_candidate_id": search["selected_candidate_id"],
        "release_gates": release_gates,
        "subgroup_eligibility": (
            None if subgroup_result is None else {
                family: value["eligible"]
                for family, value in subgroup_result["families"].items()
            }
        ),
        "customer_models": customer_model_result,
        "docker_validation": "blocked_not_run",
        "execution_assessment": {
            "run_scope": (
                "benchmark_search"
                if config.resources.benchmark_mode else "configured_search"
            ),
            "configured_candidate_count": config.tuning.iterations,
            "completed_candidate_count": sum(
                item["status"] == "succeeded" for item in search["candidate_results"]
            ),
            "cv_folds": len(split_indexes),
            "temporal_evaluation": {
                "status": (
                    "not_run"
                    if not config.splits.run_temporal_test
                    else (
                        "completed"
                        if temporal_metrics.get("available")
                        else "unavailable"
                    )
                ),
                "reason": temporal_metrics.get("reason"),
            },
            "customer_isolation": {
                "status": customer_evaluation_status,
                "eligible_customers": eligible_customer_count,
                "labeled_customers": len(customer_entries),
                "feature_snapshot_customers": len(feature_customer_names),
                "missing_label_customers": missing_label_customers,
            },
            "production_eligible": not development_only_input,
        },
    }
    write_json_atomic(run_dir / "run_result.json", run_result)
    if config.reporting.generate_html:
        generate_report(
            run_dir, run_result, holdout_metrics,
            search["candidate_results"], release_gates
        )
    return run_dir


def compare_bundle(
    bundle_dir: Path,
    feature_path: Path,
    feature_manifest_path: Path,
    label_path: Path,
    label_manifest_path: Path,
    output_dir: Path,
) -> Path:
    verification = verify_bundle(bundle_dir)
    schema = verification["schema"]
    card = verification["model_card"]
    features, _ = load_feature_snapshot(feature_path, feature_manifest_path)
    labels, _ = load_label_snapshot(label_path, label_manifest_path)
    joined, reconciliation = join_training_data(features, labels)
    matrix = numeric_matrix(joined, schema["ordered_features"])
    pipeline = joblib.load(Path(bundle_dir) / "schedule_risk_model.joblib")
    probabilities = pipeline.predict_proba(matrix)
    predictions = pipeline.predict(matrix)
    weight = float(card["significant_delay_weight"])
    metrics = classification_metrics(
        joined[LABEL_COLUMN].to_numpy(dtype=int),
        predictions,
        probabilities,
        weight,
    )
    reference = json.loads(
        (Path(bundle_dir) / "reference_metrics.json").read_text(encoding="utf-8")
    )
    baseline = reference["locked_holdout"]["overall"]
    deltas = {}
    for name, value in metrics["overall"].items():
        baseline_value = baseline.get(name)
        if isinstance(value, (int, float)) and isinstance(baseline_value, (int, float)):
            deltas[name] = {
                "current": value,
                "reference_locked_holdout": baseline_value,
                "absolute_delta": value - baseline_value,
                "relative_delta": (
                    (value - baseline_value) / abs(baseline_value)
                    if baseline_value != 0 else None
                ),
            }
    histograms = json.loads(
        (Path(bundle_dir) / "histogram_definitions.json").read_text(encoding="utf-8")
    )
    drift_rows = []
    for name in schema["ordered_features"]:
        values = pd.to_numeric(matrix[name], errors="coerce").to_numpy(dtype=float)
        drift_rows.append({
            "feature_name": name,
            "missing_rate": float(np.isnan(values).mean()),
            "population_stability_index": population_stability_index(
                values, histograms["features"][name]
            ),
        })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "metrics.json", metrics)
    write_json_atomic(output_dir / "comparison_to_training_reference.json", deltas)
    write_json_atomic(output_dir / "reconciliation.json", reconciliation)
    pd.DataFrame(drift_rows).to_parquet(output_dir / "feature_drift.parquet", index=False)
    return output_dir
