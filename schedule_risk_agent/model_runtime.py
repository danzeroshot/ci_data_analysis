from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

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

