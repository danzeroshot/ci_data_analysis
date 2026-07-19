from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from schedule_risk_agent.training_pipeline.configuration import TrainingRunConfig
from schedule_risk_agent.training_pipeline.feature_qualification import qualify_features
from schedule_risk_agent.training_pipeline.metrics import selection_score
from schedule_risk_agent.training_pipeline.splits import assign_locked_holdout, temporal_split


def test_selection_weight_endpoints():
    assert selection_score(0.6, 0.9, 0.0) == pytest.approx(0.6)
    assert selection_score(0.6, 0.9, 1.0) == pytest.approx(0.9)
    assert selection_score(0.6, 0.9, 0.5) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        selection_score(0.6, 0.9, 1.1)


def test_hash_split_is_stable_across_row_order():
    frame = pd.DataFrame({
        "CUSTOMERNAME": ["A"] * 60,
        "PROJECTID": [str(value) for value in range(60)],
        "SCHEDULERISKBIN": np.tile([0, 1, 2], 20),
    })
    first = assign_locked_holdout(frame, "v1", 0.2, 42, 1)
    reordered = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    second = assign_locked_holdout(reordered, "v1", 0.2, 42, 1)
    left = first.set_index(["CUSTOMERNAME", "PROJECTID"])["POPULATION"].sort_index()
    right = second.set_index(["CUSTOMERNAME", "PROJECTID"])["POPULATION"].sort_index()
    pd.testing.assert_series_equal(left, right)


def test_temporal_split_reports_population_counts_and_missing_dates():
    frame = pd.DataFrame({
        "PLANNEDSTARTDATE": [
            "2020-01-01", "2021-01-01", "2022-01-01",
            "2023-01-01", "2024-01-01", None,
        ],
        "SCHEDULERISKBIN": [0, 1, 2, 0, 2, 1],
    })
    result = temporal_split(frame, "PLANNEDSTARTDATE", 0.4)
    assert result["available"] is True
    assert result["train_rows"] == 3
    assert result["test_rows"] == 2
    assert result["excluded_missing_date_count"] == 1
    assert result["boundary_date"].startswith("2023-01-01")
    assert result["train_class_counts"] == {"0": 1, "1": 1, "2": 1}
    assert result["test_class_counts"] == {"0": 1, "1": 0, "2": 1}
    assert set(result["train_index"]).isdisjoint(set(result["test_index"]))

def test_qualification_rejects_leakage_constant_and_duplicate():
    frame = pd.DataFrame({
        "SAFE": [1.0, 2.0, 3.0, 4.0],
        "DUP": [1.0, 2.0, 3.0, 4.0],
        "CONSTANT": [1.0] * 4,
        "TARGETX": [0.0, 1.0, 2.0, 0.0],
    })
    manifest = {"features": [
        {"name": name, "approved": True, "inference_schema_required": True}
        for name in frame.columns
    ]}
    accepted, detail = qualify_features(
        frame, manifest, 1.0, 1, True, True
    )
    assert accepted == ["SAFE"]
    reasons = detail.set_index("feature_name")["rejection_reason"].to_dict()
    assert reasons["DUP"] == "duplicate_feature_vector"
    assert reasons["CONSTANT"] == "zero_variance"
    assert reasons["TARGETX"] == "prohibited_feature"


def test_configuration_requires_explicit_valid_weight(synthetic_training_inputs):
    raw = __import__("json").loads(
        synthetic_training_inputs["config"].read_text(encoding="utf-8")
    )
    raw["selection"]["significant_delay_weight"] = -0.1
    with pytest.raises(Exception):
        TrainingRunConfig.model_validate(raw)
