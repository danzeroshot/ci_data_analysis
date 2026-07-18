import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from schedule_risk_agent.feature_schema import FeatureSchema
from schedule_risk_agent.feature_store import LocalFeatureRepository
from schedule_risk_agent.model_runtime import ModelRuntime
from schedule_risk_agent.train import select_features


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_snapshot(tmp_path):
    root = tmp_path / "features"
    snapshot = root / "snapshots" / "build-1"
    snapshot.mkdir(parents=True)
    frame = pd.DataFrame({
        "CUSTOMERNAME": ["UDOT", "Lincoln"],
        "PROJECTID": ["1", "2"],
        "FEATUREASOFUTC": [pd.Timestamp.now(tz="UTC")] * 2,
        "FEATURE_A": [1.0, 2.0],
        "FEATURE_B": [0.0, None],
    })
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), snapshot / "features.parquet")
    manifest = {
        "build_id": "build-1",
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": "v1",
        "keyword_manifest_version": "k1",
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (snapshot / "validation.json").write_text("{}", encoding="utf-8")
    checks = {
        name: sha256(snapshot / name)
        for name in ["features.parquet", "manifest.json", "validation.json"]
    }
    (snapshot / "checksums.sha256").write_text(
        "".join("{}  {}\n".format(value, name) for name, value in checks.items()),
        encoding="utf-8",
    )
    (snapshot / "COMPLETE").write_text("", encoding="ascii")
    (root / "current.json").write_text(json.dumps({
        "build_id": "build-1",
        "relative_path": "snapshots/build-1",
    }), encoding="utf-8")
    return root


def test_local_repository_fetches_and_reports_missing(tmp_path):
    schema = FeatureSchema("v1", "k1", ["FEATURE_A", "FEATURE_B"], ["a", "b", "c"])
    repository = LocalFeatureRepository(make_snapshot(tmp_path), schema)
    repository.open()
    frame, missing, metadata = repository.fetch([("UDOT", "1"), ("UDOT", "missing")])
    assert len(frame) == 1
    assert missing == [("UDOT", "missing")]
    assert metadata["build_id"] == "build-1"


def test_local_repository_rejects_path_traversal(tmp_path):
    root = tmp_path / "features"
    root.mkdir()
    (root / "current.json").write_text(json.dumps({
        "build_id": "bad", "relative_path": "../outside"
    }), encoding="utf-8")
    schema = FeatureSchema("v1", "k1", [], ["a", "b", "c"])
    with pytest.raises(Exception):
        LocalFeatureRepository(root, schema).open()


def test_before_feature_selection_excludes_targets_and_customer():
    frame = pd.DataFrame({
        "CUSTOMERNAME": ["x"], "PERCENTDELAYED": [1.0],
        "TARGETX": [1.0], "SAFE": [2.0], "MAYBE": [3.0],
        "PROJ_KW_ROAD_COUNT": [1.0],
    })
    dictionary = pd.DataFrame({
        "FieldName": ["TARGETX", "SAFE", "MAYBE"],
        "FieldGroup": ["target/post-payment", "planned", "planned"],
        "BeginningAvailable": ["No", "Yes", "Maybe"],
    })
    assert select_features(frame, dictionary) == ["PROJ_KW_ROAD_COUNT", "SAFE"]


def test_model_runtime_probability_contract(tmp_path):
    schema = FeatureSchema("v1", "k1", ["FEATURE_A"], ["no_delay", "mild_delay", "significant_delay"])
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", DummyClassifier(strategy="prior")),
    ])
    pipeline.fit(pd.DataFrame({"FEATURE_A": [0, 1, 2]}), [0, 1, 2])
    model_path = tmp_path / "model.joblib"
    card_path = tmp_path / "card.json"
    joblib.dump(pipeline, model_path)
    card_path.write_text(json.dumps({"model_version": "test"}), encoding="utf-8")
    result = ModelRuntime(model_path, card_path, schema).predict(
        pd.DataFrame({"FEATURE_A": [1.0]})
    )[0]
    assert set(result["class_probabilities"]) == {
        "no_delay", "mild_delay", "significant_delay"
    }
    assert sum(result["class_probabilities"].values()) == pytest.approx(1.0)

