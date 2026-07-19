from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_schema import FeatureSchema


class ModelRuntime:
    def __init__(self, model_path: Path, model_card_path: Path, schema: FeatureSchema):
        self.pipeline = joblib.load(model_path)
        self.model_card = json.loads(model_card_path.read_text(encoding="utf-8"))
        self.schema = schema
        model = self.pipeline.named_steps.get("classifier", self.pipeline)
        self.classes = [int(value) for value in model.classes_]
        if self.classes != [0, 1, 2]:
            raise ValueError("Model classes must be [0, 1, 2]")

    @classmethod
    def from_bundle(cls, bundle_path: Path) -> "ModelRuntime":
        from .training_pipeline.release import verify_bundle

        bundle_path = Path(bundle_path)
        verification = verify_bundle(bundle_path)
        card = verification["model_card"]
        if card.get("status") == "rejected":
            raise ValueError("Rejected model bundle cannot be loaded")
        schema = FeatureSchema.load(bundle_path / "schedule_risk_feature_schema.json")
        runtime = cls(
            bundle_path / "schedule_risk_model.joblib",
            bundle_path / "schedule_risk_model_card.json",
            schema,
        )
        parity_input = pd.read_parquet(bundle_path / "parity_input.parquet")
        expected = json.loads(
            (bundle_path / "parity_expected_predictions.json").read_text(encoding="utf-8")
        )
        probability = runtime.pipeline.predict_proba(parity_input)
        prediction = runtime.pipeline.predict(parity_input)
        tolerance = float(expected.get("absolute_tolerance", 1e-12))
        if not np.array_equal(prediction, np.asarray(expected["predictions"], dtype=int)):
            raise ValueError("Model bundle parity predictions do not match")
        if not np.allclose(
            probability,
            np.asarray(expected["probabilities"], dtype=float),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("Model bundle parity probabilities do not match")
        runtime.bundle_path = bundle_path
        return runtime

    def predict(self, frame):
        matrix = frame.reindex(columns=self.schema.ordered_features)
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        probabilities = self.pipeline.predict_proba(matrix)
        selected_indexes = probabilities.argmax(axis=1)
        labels = self.schema.class_labels
        results = []
        for index, selected_index in enumerate(selected_indexes):
            class_id = self.classes[selected_index]
            mapping = {
                labels[position]: float(probabilities[index, position])
                for position in range(len(labels))
            }
            results.append({
                "risk_bin": class_id,
                "risk_label": labels[class_id],
                "predicted_class_probability": float(probabilities[index, selected_index]),
                "class_probabilities": mapping,
            })
        return results

