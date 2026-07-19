from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .contracts import CLASS_IDS, DataContractError, LABEL_COLUMN


def stable_hash_fraction(
    target_version: str,
    customer: str,
    project_id: str,
    seed: int,
) -> float:
    canonical = "{}|{}|{}|{}".format(target_version, customer, project_id, seed)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2 ** 64)


def assign_locked_holdout(
    frame: pd.DataFrame,
    target_version: str,
    holdout_fraction: float,
    seed: int,
    minimum_class_support: int,
) -> pd.DataFrame:
    fractions = [
        stable_hash_fraction(target_version, row.CUSTOMERNAME, row.PROJECTID, seed)
        for row in frame[["CUSTOMERNAME", "PROJECTID"]].itertuples(index=False)
    ]
    assignments = frame[["CUSTOMERNAME", "PROJECTID", LABEL_COLUMN]].copy()
    assignments["HASHFRACTION"] = fractions
    assignments["POPULATION"] = np.where(
        assignments["HASHFRACTION"] < holdout_fraction,
        "locked_holdout",
        "development",
    )
    for population in ("development", "locked_holdout"):
        counts = assignments.loc[
            assignments["POPULATION"].eq(population), LABEL_COLUMN
        ].value_counts()
        missing = [
            class_id for class_id in CLASS_IDS
            if int(counts.get(class_id, 0)) < minimum_class_support
        ]
        if missing:
            raise DataContractError(
                "{} has insufficient support for classes {}".format(population, missing)
            )
    if assignments.duplicated(["CUSTOMERNAME", "PROJECTID"]).any():
        raise DataContractError("Split assignments contain duplicate project keys")
    return assignments


def make_cv_assignments(
    development: pd.DataFrame,
    folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]]]:
    labels = development[LABEL_COLUMN].to_numpy()
    counts = pd.Series(labels).value_counts()
    maximum_folds = int(counts.min())
    if maximum_folds < 2:
        raise DataContractError("Development data cannot support stratified cross-validation")
    actual_folds = min(folds, maximum_folds)
    splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    split_indexes = list(splitter.split(np.zeros(len(development)), labels))
    fold_values = np.full(len(development), -1, dtype=int)
    for fold, (_, validation_index) in enumerate(split_indexes):
        fold_values[validation_index] = fold
    assignments = development[["CUSTOMERNAME", "PROJECTID", LABEL_COLUMN]].copy()
    assignments["CVFOLD"] = fold_values
    return assignments, split_indexes


def temporal_split(
    frame: pd.DataFrame,
    date_column: str,
    test_fraction: float,
) -> Dict[str, Any]:
    if date_column not in frame:
        return {"available": False, "reason": "planned_start_date_not_available"}
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    valid_indexes = np.flatnonzero(dates.notna().to_numpy())
    if len(valid_indexes) < 2:
        return {"available": False, "reason": "insufficient_valid_planned_dates"}
    ordered = valid_indexes[np.argsort(dates.iloc[valid_indexes].to_numpy())]
    test_count = max(1, int(np.ceil(len(ordered) * test_fraction)))
    train_index = ordered[:-test_count]
    test_index = ordered[-test_count:]
    if not len(train_index):
        return {"available": False, "reason": "empty_temporal_training_population"}
    train_counts = frame.iloc[train_index][LABEL_COLUMN].value_counts()
    test_counts = frame.iloc[test_index][LABEL_COLUMN].value_counts()
    train_class_counts = {
        str(class_id): int(train_counts.get(class_id, 0)) for class_id in CLASS_IDS
    }
    test_class_counts = {
        str(class_id): int(test_counts.get(class_id, 0)) for class_id in CLASS_IDS
    }
    return {
        "available": True,
        "train_index": train_index,
        "test_index": test_index,
        "boundary_date": str(dates.iloc[test_index].min()),
        "excluded_missing_date_count": int(dates.isna().sum()),
        "train_rows": int(len(train_index)),
        "test_rows": int(len(test_index)),
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
    }


def customer_holdout_splits(
    frame: pd.DataFrame,
    minimum_rows: int,
    minimum_class_support: int,
) -> List[Dict[str, Any]]:
    results = []
    for customer in sorted(frame["CUSTOMERNAME"].unique()):
        test_mask = frame["CUSTOMERNAME"].eq(customer).to_numpy()
        test_index = np.flatnonzero(test_mask)
        train_index = np.flatnonzero(~test_mask)
        counts = frame.iloc[test_index][LABEL_COLUMN].value_counts()
        classes_with_support = sum(
            int(counts.get(class_id, 0)) >= minimum_class_support
            for class_id in CLASS_IDS
        )
        eligible = len(test_index) >= minimum_rows and classes_with_support == len(CLASS_IDS)
        results.append({
            "customer": customer,
            "eligible": bool(eligible),
            "reason": None if eligible else "insufficient_rows_or_class_support",
            "train_index": train_index,
            "test_index": test_index,
            "test_rows": int(len(test_index)),
            "class_counts": {
                str(class_id): int(counts.get(class_id, 0)) for class_id in CLASS_IDS
            },
        })
    return results
