from __future__ import annotations

import json

import pandas as pd

from schedule_risk_agent.training_pipeline.release import (
    promote_run,
    rollback_release,
    verify_bundle,
)
from schedule_risk_agent.training_pipeline.stages import compare_bundle, run_pipeline


def test_full_training_release_and_comparison(synthetic_training_inputs, tmp_path):
    run_dir = run_pipeline(synthetic_training_inputs["config"])
    result = json.loads((run_dir / "run_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "host_validated"
    assert result["accepted_feature_count"] == 3
    assessment = result["execution_assessment"]
    assert assessment["run_scope"] == "benchmark_search"
    assert assessment["completed_candidate_count"] == 2
    assert assessment["temporal_evaluation"]["status"] == "not_run"
    assert assessment["customer_isolation"]["status"] == "not_run"

    report_path = run_dir / "report.html"
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    for required_text in [
        "Development assessment, not production validation.",
        "Execution Coverage",
        "Data Coverage and Label Availability",
        "Model Selection Scope",
        "Training, Cross-Validation, and Holdout",
        "Customer-Isolation Evaluation",
        "Temporal Evaluation",
        "Feature Qualification",
        "Lineage and Reproducibility",
        "Planned subgroup evaluations",
    ]:
        assert required_text in report
    assert "data:image/png;base64," in report
    assert "src='plots/" not in report
    assert "__CARDS__" not in report

    assert (run_dir / "feature_importances.parquet").is_file()
    assert (run_dir / "candidate" / "feature_importances.parquet").is_file()
    metric_frame = pd.read_parquet(run_dir / "metrics" / "metrics_long.parquet")
    assert set(metric_frame["evaluation_population"]) == {
        "development_in_sample",
        "cross_validation_out_of_fold",
        "locked_hash_holdout",
    }
    verify_bundle(run_dir / "candidate")
    from schedule_risk_agent.model_runtime import ModelRuntime
    runtime = ModelRuntime.from_bundle(run_dir / "candidate")
    assert runtime.model_card["model_version"] == result["model_version"]

    release = promote_run(
        run_dir,
        synthetic_training_inputs["output_root"],
        production=False,
    )
    assert release.is_dir()
    rollback_release(
        synthetic_training_inputs["output_root"],
        result["model_version"],
    )

    comparison = compare_bundle(
        release,
        synthetic_training_inputs["feature_root"] / "features.parquet",
        synthetic_training_inputs["feature_root"] / "manifest.json",
        synthetic_training_inputs["label_root"] / "labels.parquet",
        synthetic_training_inputs["label_root"] / "manifest.json",
        tmp_path / "comparison",
    )
    assert (comparison / "metrics.json").is_file()
    assert (comparison / "comparison_to_training_reference.json").is_file()
    assert (comparison / "feature_drift.parquet").is_file()
