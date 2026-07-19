from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from .metrics import classification_metrics


def build_pipeline(parameters: Dict[str, Any], seed: int, n_jobs: int) -> Pipeline:
    parameters = dict(parameters)
    add_indicator = bool(parameters.pop("missing_indicators", False))
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("classifier", RandomForestClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **parameters
        )),
    ])


def parameter_space(tuning: Any) -> Dict[str, Sequence[Any]]:
    return {
        "n_estimators": tuning.n_estimators,
        "max_depth": tuning.max_depth,
        "min_samples_leaf": tuning.min_samples_leaf,
        "min_samples_split": tuning.min_samples_split,
        "max_features": tuning.max_features,
        "max_samples": tuning.max_samples,
        "class_weight": tuning.class_weight,
        "criterion": tuning.criterion,
        "missing_indicators": tuning.missing_indicators,
    }


def tune_random_forest(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    split_indexes: List[Tuple[np.ndarray, np.ndarray]],
    tuning: Any,
    weight: float,
    seed: int,
    n_jobs: int,
    calibration_bins: int,
) -> Dict[str, Any]:
    candidates = list(ParameterSampler(
        parameter_space(tuning),
        n_iter=tuning.iterations,
        random_state=seed,
    ))
    candidate_results = []
    all_oof = []
    for candidate_index, parameters in enumerate(candidates):
        candidate_id = "candidate-{:04d}".format(candidate_index + 1)
        fold_results = []
        oof_probability = np.full((len(matrix), 3), np.nan, dtype=float)
        oof_prediction = np.full(len(matrix), -1, dtype=int)
        candidate_started = time.time()
        failed = None
        for fold, (train_index, validation_index) in enumerate(split_indexes):
            pipeline = build_pipeline(parameters, seed + fold, n_jobs)
            started = time.time()
            captured_warnings = []
            try:
                with warnings.catch_warnings(record=True) as warning_records:
                    warnings.simplefilter("always")
                    pipeline.fit(matrix.iloc[train_index], labels[train_index])
                    validation_probability = pipeline.predict_proba(matrix.iloc[validation_index])
                    validation_prediction = pipeline.predict(matrix.iloc[validation_index])
                    train_probability = pipeline.predict_proba(matrix.iloc[train_index])
                    train_prediction = pipeline.predict(matrix.iloc[train_index])
                    captured_warnings = [str(item.message) for item in warning_records]
                oof_probability[validation_index] = validation_probability
                oof_prediction[validation_index] = validation_prediction
                validation_metrics = classification_metrics(
                    labels[validation_index], validation_prediction,
                    validation_probability, weight, calibration_bins
                )
                train_metrics = classification_metrics(
                    labels[train_index], train_prediction,
                    train_probability, weight, calibration_bins
                )
                fold_results.append({
                    "fold": fold,
                    "train_rows": int(len(train_index)),
                    "validation_rows": int(len(validation_index)),
                    "fit_and_score_seconds": float(time.time() - started),
                    "warnings": captured_warnings,
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                })
            except Exception as exc:
                failed = "{}: {}".format(type(exc).__name__, exc)
                break
        if failed is None:
            aggregate = classification_metrics(
                labels, oof_prediction, oof_probability, weight, calibration_bins
            )
            mean_train_macro_f1 = float(np.mean([
                fold["train_metrics"]["overall"]["macro_f1"] for fold in fold_results
            ]))
            aggregate["train_to_validation_macro_f1_gap"] = (
                mean_train_macro_f1 - aggregate["overall"]["macro_f1"]
            )
            for row_index in range(len(matrix)):
                all_oof.append({
                    "candidate_id": candidate_id,
                    "row_index": row_index,
                    "actual": int(labels[row_index]),
                    "predicted": int(oof_prediction[row_index]),
                    "probability_no_delay": float(oof_probability[row_index, 0]),
                    "probability_mild_delay": float(oof_probability[row_index, 1]),
                    "probability_significant_delay": float(oof_probability[row_index, 2]),
                })
        else:
            aggregate = None
        candidate_results.append({
            "candidate_id": candidate_id,
            "parameters": parameters,
            "status": "failed" if failed else "succeeded",
            "failure": failed,
            "elapsed_seconds": float(time.time() - candidate_started),
            "folds": fold_results,
            "aggregate": aggregate,
        })
    valid = [item for item in candidate_results if item["status"] == "succeeded"]
    if not valid:
        raise RuntimeError("All random-forest candidates failed")
    valid.sort(key=lambda item: (
        -item["aggregate"]["overall"]["selection_score"],
        -item["aggregate"]["overall"]["macro_f1"],
        -item["aggregate"]["overall"]["significant_delay_recall"],
        item["aggregate"]["train_to_validation_macro_f1_gap"],
        float("inf") if item["parameters"]["max_depth"] is None else item["parameters"]["max_depth"],
        -item["parameters"]["min_samples_leaf"],
        item["elapsed_seconds"],
        str(sorted(item["parameters"].items())),
    ))
    for rank, item in enumerate(valid, start=1):
        item["rank"] = rank
    selected = valid[0]
    return {
        "selected_candidate_id": selected["candidate_id"],
        "selected_parameters": selected["parameters"],
        "candidate_results": candidate_results,
        "ranked_candidate_ids": [item["candidate_id"] for item in valid],
        "oof_predictions": pd.DataFrame(all_oof),
    }
