from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from schedule_risk_agent.training_pipeline.lineage import write_checksums, write_json_atomic


@pytest.fixture
def synthetic_training_inputs(tmp_path: Path) -> Dict[str, Path]:
    rng = np.random.RandomState(12)
    rows = 150
    customers = np.array(["A", "B", "C"])
    customer = customers[np.arange(rows) % len(customers)]
    project_id = np.arange(1, rows + 1).astype(str)
    signal = rng.normal(size=rows)
    noise = rng.normal(size=rows)
    labels = np.where(signal < -0.35, 0, np.where(signal < 0.45, 1, 2)).astype(int)
    percent = np.where(labels == 0, -10 + signal, np.where(labels == 1, 12 + signal, 45 + signal))

    feature_root = tmp_path / "features"
    feature_root.mkdir()
    feature_frame = pd.DataFrame({
        "CUSTOMERNAME": customer,
        "PROJECTID": project_id,
        "FEATUREASOFUTC": pd.Timestamp("2026-01-01", tz="UTC"),
        "SIGNAL": signal,
        "NOISE": noise,
        "WITH_MISSING": np.where(np.arange(rows) % 7 == 0, np.nan, signal * 0.5),
        "DUPLICATE_SIGNAL": signal,
        "CONSTANT": 1.0,
        "TARGETLEAK": labels,
    })
    feature_frame.to_parquet(feature_root / "features.parquet", index=False)
    feature_manifest = {
        "schema_version": "test-features-v1",
        "build_id": "test-features",
        "row_count": rows,
        "column_count": len(feature_frame.columns),
        "feature_schema_version": "test-feature-schema-v1",
        "keyword_manifest_version": "test-keywords-v1",
    }
    write_json_atomic(feature_root / "manifest.json", feature_manifest)
    write_checksums(feature_root, ["features.parquet", "manifest.json"])
    (feature_root / "COMPLETE").write_text("", encoding="ascii")

    label_root = tmp_path / "labels"
    label_root.mkdir()
    label_frame = pd.DataFrame({
        "CUSTOMERNAME": customer,
        "PROJECTID": project_id,
        "PERCENTDELAYED": percent,
        "SCHEDULERISKBIN": labels,
    })
    label_frame.to_parquet(label_root / "labels.parquet", index=False)
    label_manifest = {
        "schema_version": "test-labels-v1",
        "build_id": "test-labels",
        "row_count": rows,
        "target_definition_version": "schedule-delay-v1",
        "development_only": False,
    }
    write_json_atomic(label_root / "manifest.json", label_manifest)
    write_checksums(label_root, ["labels.parquet", "manifest.json"])
    (label_root / "COMPLETE").write_text("", encoding="ascii")

    candidate_manifest = tmp_path / "candidate_features.json"
    write_json_atomic(candidate_manifest, {
        "schema_version": "schedule-candidate-features-v1",
        "feature_schema_version": "test-feature-schema-v1",
        "keyword_manifest_version": "test-keywords-v1",
        "features": [
            {"name": name, "family": "test", "source": "fixture",
             "expected_type": "numeric", "approved": True,
             "inference_schema_required": True}
            for name in [
                "SIGNAL", "NOISE", "WITH_MISSING", "DUPLICATE_SIGNAL",
                "CONSTANT", "TARGETLEAK"
            ]
        ],
    })
    release_policy = tmp_path / "release_policy.json"
    write_json_atomic(release_policy, {
        "schema_version": "schedule-release-policy-v1",
        "metric_thresholds": {
            "minimum_macro_f1": None,
            "minimum_significant_delay_recall": None,
            "minimum_balanced_accuracy": None,
            "maximum_calibration_error": None,
        },
    })
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "run.json"
    write_json_atomic(config_path, {
        "schema_version": "schedule-training-run-v1",
        "run_name": "fixture",
        "random_seed": 42,
        "training_snapshot": {
            "path": str(feature_root / "features.parquet"),
            "manifest_path": str(feature_root / "manifest.json"),
        },
        "labels": {
            "path": str(label_root / "labels.parquet"),
            "manifest_path": str(label_root / "manifest.json"),
        },
        "target_definition_version": "schedule-delay-v1",
        "feature_policy": {
            "manifest_path": str(candidate_manifest),
            "maximum_missing_rate": 0.95,
            "minimum_non_null_count": 20,
            "drop_zero_variance": True,
            "drop_duplicate_vectors": True,
        },
        "selection": {
            "significant_delay_weight": 0.4,
            "primary_metric": "weighted_macro_f1_significant_recall",
        },
        "tuning": {
            "strategy": "randomized_search",
            "iterations": 2,
            "cross_validation_folds": 3,
            "n_estimators": [20],
            "max_depth": [3, 5],
            "min_samples_leaf": [3, 5],
            "min_samples_split": [6],
            "max_features": ["sqrt"],
            "max_samples": [0.8],
            "class_weight": ["balanced_subsample"],
            "criterion": ["gini"],
            "missing_indicators": [False],
        },
        "splits": {
            "locked_holdout_fraction": 0.2,
            "temporal_test_fraction": 0.2,
            "split_seed": 42,
            "minimum_class_support": 2,
            "minimum_customer_rows": 10,
            "minimum_customer_class_support": 2,
            "run_temporal_test": False,
            "temporal_date_column": "PLANNEDSTARTDATE",
            "run_customer_tests": False,
        },
        "resources": {
            "model_n_jobs": 1,
            "maximum_concurrent_candidates": 1,
            "bootstrap_iterations": 20,
            "benchmark_mode": True,
        },
        "reporting": {
            "generate_html": True,
            "top_candidate_count": 20,
            "calibration_bins": 5,
        },
        "release_policy_path": str(release_policy),
        "output_root": str(output_root),
    })
    return {
        "config": config_path,
        "feature_root": feature_root,
        "label_root": label_root,
        "output_root": output_root,
    }
