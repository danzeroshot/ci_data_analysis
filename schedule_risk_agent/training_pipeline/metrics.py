from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .contracts import CLASS_IDS, CLASS_LABELS


def selection_score(macro_f1: float, significant_recall: float, weight: float) -> float:
    if weight < 0.0 or weight > 1.0:
        raise ValueError("weight must be between 0 and 1")
    return (1.0 - weight) * macro_f1 + weight * significant_recall


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(len(CLASS_IDS), dtype=float)[y_true.astype(int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def calibration_errors(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> Dict[str, float]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions.eq(y_true) if hasattr(predictions, "eq") else predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    expected = 0.0
    maximum = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not mask.any():
            continue
        gap = abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
        expected += float(mask.mean()) * gap
        maximum = max(maximum, gap)
    return {
        "expected_calibration_error": float(expected),
        "maximum_calibration_error": float(maximum),
    }


def _safe_auc(y_binary: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    if len(np.unique(y_binary)) < 2:
        return {"value": None, "undefined_reason": "class_absent_in_actual"}
    return {"value": float(roc_auc_score(y_binary, score)), "undefined_reason": None}


def classification_metrics(
    y_true: Sequence[int],
    predictions: Sequence[int],
    probabilities: np.ndarray,
    significant_delay_weight: float,
    calibration_bins: int = 10,
) -> Dict[str, Any]:
    y_true_array = np.asarray(y_true, dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y_true_array), len(CLASS_IDS)):
        raise ValueError("probability matrix must have three columns")
    macro_f1 = float(f1_score(
        y_true_array, prediction_array, labels=CLASS_IDS, average="macro", zero_division=0
    ))
    significant_recall = float(recall_score(
        y_true_array, prediction_array, labels=[2], average="macro", zero_division=0
    ))
    overall = {
        "sample_count": int(len(y_true_array)),
        "accuracy": float(accuracy_score(y_true_array, prediction_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, prediction_array)),
        "macro_precision": float(precision_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="macro", zero_division=0
        )),
        "weighted_precision": float(precision_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="weighted", zero_division=0
        )),
        "micro_precision": float(precision_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="micro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="macro", zero_division=0
        )),
        "weighted_recall": float(recall_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="weighted", zero_division=0
        )),
        "micro_recall": float(recall_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="micro", zero_division=0
        )),
        "macro_f1": macro_f1,
        "weighted_f1": float(f1_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="weighted", zero_division=0
        )),
        "micro_f1": float(f1_score(
            y_true_array, prediction_array, labels=CLASS_IDS, average="micro", zero_division=0
        )),
        "significant_delay_recall": significant_recall,
        "selection_score": selection_score(
            macro_f1, significant_recall, significant_delay_weight
        ),
        "log_loss": float(log_loss(y_true_array, probabilities, labels=CLASS_IDS)),
        "multiclass_brier": multiclass_brier(y_true_array, probabilities),
    }
    try:
        overall["roc_auc_ovr_macro"] = float(roc_auc_score(
            y_true_array, probabilities, labels=CLASS_IDS, multi_class="ovr", average="macro"
        ))
        overall["roc_auc_ovr_weighted"] = float(roc_auc_score(
            y_true_array, probabilities, labels=CLASS_IDS, multi_class="ovr", average="weighted"
        ))
        overall["roc_auc_ovo_macro"] = float(roc_auc_score(
            y_true_array, probabilities, labels=CLASS_IDS, multi_class="ovo", average="macro"
        ))
    except ValueError:
        overall["roc_auc_ovr_macro"] = None
        overall["roc_auc_ovr_weighted"] = None
        overall["roc_auc_ovo_macro"] = None
    overall.update(calibration_errors(
        y_true_array, probabilities, bins=calibration_bins
    ))

    matrix = confusion_matrix(y_true_array, prediction_array, labels=CLASS_IDS)
    per_class = {}
    for class_id, label in zip(CLASS_IDS, CLASS_LABELS):
        tp = int(matrix[class_id, class_id])
        fn = int(matrix[class_id, :].sum() - tp)
        fp = int(matrix[:, class_id].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        support = tp + fn
        predicted_count = tp + fp
        precision = tp / predicted_count if predicted_count else None
        recall = tp / support if support else None
        specificity = tn / (tn + fp) if (tn + fp) else None
        class_f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        binary = (y_true_array == class_id).astype(int)
        auc = _safe_auc(binary, probabilities[:, class_id])
        average_precision = (
            float(average_precision_score(binary, probabilities[:, class_id]))
            if len(np.unique(binary)) > 1 else None
        )
        calibration_intercept = calibration_slope = None
        if len(np.unique(binary)) > 1 and len(binary) >= 20:
            clipped = np.clip(probabilities[:, class_id], 1e-6, 1 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            try:
                calibration = LogisticRegression(
                    penalty=None, solver="lbfgs", max_iter=1000
                ).fit(logits, binary)
                calibration_intercept = float(calibration.intercept_[0])
                calibration_slope = float(calibration.coef_[0, 0])
            except Exception:
                pass
        per_class[label] = {
            "class_id": class_id,
            "support": support,
            "prevalence": float(support / len(y_true_array)) if len(y_true_array) else None,
            "predicted_count": predicted_count,
            "predicted_proportion": (
                float(predicted_count / len(y_true_array)) if len(y_true_array) else None
            ),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": class_f1,
            "false_negative_rate": (fn / support) if support else None,
            "false_positive_rate": (fp / (fp + tn)) if (fp + tn) else None,
            "roc_auc_ovr": auc["value"],
            "roc_auc_undefined_reason": auc["undefined_reason"],
            "average_precision": average_precision,
            "brier": float(np.mean((probabilities[:, class_id] - binary) ** 2)),
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
        }
    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix_count": matrix.tolist(),
        "confusion_matrix_actual_normalized": np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix, dtype=float),
            where=matrix.sum(axis=1, keepdims=True) != 0,
        ).tolist(),
        "confusion_matrix_predicted_normalized": np.divide(
            matrix,
            matrix.sum(axis=0, keepdims=True),
            out=np.zeros_like(matrix, dtype=float),
            where=matrix.sum(axis=0, keepdims=True) != 0,
        ).tolist(),
    }


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    predictions: Sequence[int],
    probabilities: np.ndarray,
    weight: float,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(predictions, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    rng = np.random.RandomState(seed)
    indexes_by_class = {class_id: np.flatnonzero(y == class_id) for class_id in CLASS_IDS}
    values = {"macro_f1": [], "balanced_accuracy": [], "significant_delay_recall": [], "selection_score": []}
    valid = 0
    for _ in range(iterations):
        parts = []
        for class_id in CLASS_IDS:
            indexes = indexes_by_class[class_id]
            if not len(indexes):
                parts = []
                break
            parts.append(rng.choice(indexes, size=len(indexes), replace=True))
        if not parts:
            continue
        sampled = np.concatenate(parts)
        result = classification_metrics(y[sampled], pred[sampled], prob[sampled], weight)
        for name in values:
            values[name].append(result["overall"][name])
        valid += 1
    output = {"attempted_iterations": iterations, "valid_iterations": valid, "metrics": {}}
    for name, samples in values.items():
        output["metrics"][name] = {
            "estimate": float(classification_metrics(y, pred, prob, weight)["overall"][name]),
            "lower_95": float(np.quantile(samples, 0.025)) if samples else None,
            "upper_95": float(np.quantile(samples, 0.975)) if samples else None,
        }
    return output
