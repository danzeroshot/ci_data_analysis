from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import CLASS_IDS, CLASS_LABELS


QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _histogram(values: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.array([0.0, 1.0]), np.array([0])
    unique = np.unique(finite)
    if len(unique) == 1:
        value = float(unique[0])
        width = max(abs(value) * 0.01, 1e-6)
        edges = np.array([value - width, value + width], dtype=float)
    else:
        candidate = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(candidate)
        if len(edges) < 2:
            edges = np.array([float(finite.min()), float(finite.max()) + 1e-6])
        edges[0] = -np.inf
        edges[-1] = np.inf
    counts, edges = np.histogram(finite, bins=edges)
    return edges, counts


def feature_reference_profile(
    matrix: pd.DataFrame,
    bins: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records = []
    histograms = {}
    for name in matrix.columns:
        numeric = pd.to_numeric(matrix[name], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        edges, counts = _histogram(values, bins)
        quantiles = {
            str(q): float(np.quantile(finite, q)) if len(finite) else None
            for q in QUANTILES
        }
        q1 = quantiles["0.25"]
        q3 = quantiles["0.75"]
        iqr = (q3 - q1) if q1 is not None and q3 is not None else None
        records.append({
            "feature_name": name,
            "dtype": str(matrix[name].dtype),
            "row_count": int(len(values)),
            "non_null_count": int(numeric.notna().sum()),
            "missing_rate": float(numeric.isna().mean()),
            "finite_count": int(len(finite)),
            "minimum": float(np.min(finite)) if len(finite) else None,
            "maximum": float(np.max(finite)) if len(finite) else None,
            "mean": float(np.mean(finite)) if len(finite) else None,
            "standard_deviation": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "zero_proportion": float(np.mean(finite == 0)) if len(finite) else None,
            "lower_outlier_bound": (q1 - 1.5 * iqr) if iqr is not None else None,
            "upper_outlier_bound": (q3 + 1.5 * iqr) if iqr is not None else None,
            **{"quantile_{}".format(str(q).replace(".", "_")): quantiles[str(q)] for q in QUANTILES},
        })
        histograms[name] = {
            "edges": [
                "-Infinity" if np.isneginf(value) else
                "Infinity" if np.isposinf(value) else float(value)
                for value in edges
            ],
            "counts": [int(value) for value in counts],
        }
    return pd.DataFrame(records), {
        "schema_version": "schedule-feature-histograms-v1",
        "binning": "training_quantile_bins",
        "features": histograms,
    }


def prediction_reference_profile(
    y_true: Sequence[int],
    predictions: Sequence[int],
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    confidence = probabilities.max(axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    quantile_points = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

    def quantile_map(values: np.ndarray) -> Dict[str, float]:
        return {str(q): float(np.quantile(values, q)) for q in quantile_points}

    by_actual = {}
    for class_id, label in zip(CLASS_IDS, CLASS_LABELS):
        mask = y == class_id
        by_actual[label] = {
            "rows": int(mask.sum()),
            "probability_quantiles": {
                CLASS_LABELS[index]: quantile_map(probabilities[mask, index])
                if mask.any() else None
                for index in CLASS_IDS
            },
        }
    return {
        "schema_version": "schedule-prediction-reference-v1",
        "actual_class_counts": {
            CLASS_LABELS[class_id]: int(np.sum(y == class_id)) for class_id in CLASS_IDS
        },
        "predicted_class_counts": {
            CLASS_LABELS[class_id]: int(np.sum(predicted == class_id)) for class_id in CLASS_IDS
        },
        "confidence_quantiles": quantile_map(confidence),
        "margin_quantiles": quantile_map(margin),
        "by_actual_class": by_actual,
    }


def population_stability_index(
    values: np.ndarray,
    histogram: Dict[str, Any],
    epsilon: float = 1e-6,
) -> float:
    converted_edges = []
    for value in histogram["edges"]:
        if value == "-Infinity":
            converted_edges.append(-np.inf)
        elif value == "Infinity":
            converted_edges.append(np.inf)
        else:
            converted_edges.append(float(value))
    actual_counts, _ = np.histogram(
        np.asarray(values, dtype=float)[np.isfinite(values)],
        bins=np.asarray(converted_edges, dtype=float),
    )
    expected_counts = np.asarray(histogram["counts"], dtype=float)
    expected = (expected_counts + epsilon) / (expected_counts.sum() + epsilon * len(expected_counts))
    actual = (actual_counts + epsilon) / (actual_counts.sum() + epsilon * len(actual_counts))
    return float(np.sum((actual - expected) * np.log(actual / expected)))
