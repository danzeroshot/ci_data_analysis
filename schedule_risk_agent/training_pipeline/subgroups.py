from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .contracts import CLASS_IDS, LABEL_COLUMN
from .metrics import classification_metrics


FAMILY_NAMES = (
    "planned_duration",
    "planned_value",
    "contract_item_count",
    "predictor_missingness",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _assign(series: pd.Series, rules: List[Tuple[str, Any]], invalid: pd.Series):
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    valid = ~invalid
    for band, mask in rules:
        result.loc[valid & mask] = band
    overlap = result.notna() & result.duplicated(keep=False)
    if overlap.any():
        raise ValueError("Overlapping subgroup band assignment")
    return result


def assign_family(frame: pd.DataFrame, family: str, config: Any,
                  accepted_features: Iterable[str] = ()) -> Tuple[pd.Series, Dict[str, Any]]:
    if family == "planned_duration":
        source = "PLANNEDDURATIONDAYS"
        if source not in frame:
            return pd.Series(pd.NA, index=frame.index, dtype="string"), {
                "family": family, "source": source, "status": "fail",
                "reason": "source_field_missing",
            }
        values = _numeric(frame[source])
        invalid = values.isna() | (values <= 0)
        b = config.duration_boundaries_days
        rules = [
            ("duration_lt_60", values < b[0]),
            ("duration_60_to_364", (values >= b[0]) & (values < b[1])),
            ("duration_ge_365", values >= b[1]),
        ]
    elif family == "planned_value":
        source = "PROJECTPLANNEDVALUE"
        if source not in frame:
            return pd.Series(pd.NA, index=frame.index, dtype="string"), {
                "family": family, "source": source, "status": "fail",
                "reason": "source_field_missing",
            }
        values = _numeric(frame[source])
        invalid = values.isna()
        b = config.planned_value_boundaries
        rules = [
            ("value_le_0", values <= b[0]),
            ("value_0_to_1m", (values > b[0]) & (values < b[1])),
            ("value_1m_to_10m", (values >= b[1]) & (values < b[2])),
            ("value_ge_10m", values >= b[2]),
        ]
    elif family == "contract_item_count":
        source = "NUMCONTRACTITEMS"
        if source not in frame:
            return pd.Series(pd.NA, index=frame.index, dtype="string"), {
                "family": family, "source": source, "status": "fail",
                "reason": "source_field_missing",
            }
        values = _numeric(frame[source])
        invalid = values.isna() | (values < 1) | (values % 1 != 0)
        b = config.contract_item_boundaries
        rules = [
            ("items_1_to_19", (values >= 1) & (values < b[0])),
            ("items_20_to_49", (values >= b[0]) & (values < b[1])),
            ("items_ge_50", values >= b[1]),
        ]
    elif family == "predictor_missingness":
        names = list(accepted_features)
        if not names or any(name not in frame for name in names):
            return pd.Series(pd.NA, index=frame.index, dtype="string"), {
                "family": family, "source": "qualified_feature_schema", "status": "fail",
                "reason": "qualified_schema_missing",
            }
        matrix = frame[names].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        values = matrix.isna().mean(axis=1)
        source = "qualified_feature_schema"
        invalid = values.isna()
        rules = [
            ("missingness_zero", values == 0),
            ("missingness_nonzero", values > 0),
        ]
    else:
        raise ValueError("Unknown subgroup family: {}".format(family))
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for band, mask in rules:
        result.loc[~invalid & mask] = band
    return result, {
        "family": family,
        "source": source,
        "status": "pass" if not result.isna().any() else "fail",
        "source_missing_count": int(result.isna().sum()),
        "source_value_min": float(values.min()) if values.notna().any() else None,
        "source_value_max": float(values.max()) if values.notna().any() else None,
    }


def _configured_bands(family: str, config: Any) -> List[str]:
    if family == "planned_duration":
        return ["duration_lt_60", "duration_60_to_364", "duration_ge_365"]
    if family == "planned_value":
        return ["value_le_0", "value_0_to_1m", "value_1m_to_10m", "value_ge_10m"]
    if family == "contract_item_count":
        return ["items_1_to_19", "items_20_to_49", "items_ge_50"]
    if family == "predictor_missingness":
        return ["missingness_zero", "missingness_nonzero"]
    raise ValueError("Unknown subgroup family: {}".format(family))


def _support(rows: pd.DataFrame, population: str, family: str, bands: List[str],
             total_min: int, class_min: int) -> List[Dict[str, Any]]:
    output = []
    for band in bands:
        subset = rows[rows["BAND"] == band]
        counts = subset[LABEL_COLUMN].value_counts()
        class_counts = {str(c): int(counts.get(c, 0)) for c in CLASS_IDS}
        failed = []
        if len(subset) < total_min:
            failed.append("minimum_rows")
        if any(class_counts[str(c)] < class_min for c in CLASS_IDS):
            failed.append("minimum_class_support")
        output.append({
            "population": population,
            "band": band,
            "rows": int(len(subset)),
            "class_counts": class_counts,
            "minimum_rows": total_min,
            "minimum_rows_per_class": class_min,
            "status": "pass" if not failed else "fail",
            "failed_criteria": failed,
        })
    return output


def evaluate_subgroups(joined: pd.DataFrame, development_mask: np.ndarray,
                       accepted_features: Iterable[str], oof_predictions: pd.DataFrame,
                       holdout_predictions: pd.DataFrame, config: Any,
                       weight: float, calibration_bins: int) -> Dict[str, Any]:
    families = {}
    all_assignments = []
    for family in FAMILY_NAMES:
        dev = joined.loc[development_mask].reset_index(drop=True)
        hold = joined.loc[~development_mask].reset_index(drop=True)
        dev_band, source_detail = assign_family(dev, family, config, accepted_features)
        hold_band, hold_detail = assign_family(hold, family, config, accepted_features)
        dev_rows = dev[["CUSTOMERNAME", "PROJECTID", LABEL_COLUMN]].copy()
        hold_rows = hold[["CUSTOMERNAME", "PROJECTID", LABEL_COLUMN]].copy()
        dev_rows["POPULATION"] = "cv_oof"
        hold_rows["POPULATION"] = "locked_holdout"
        dev_rows["BAND"] = dev_band.astype("string")
        hold_rows["BAND"] = hold_band.astype("string")
        combined = pd.concat([dev_rows, hold_rows], ignore_index=True)
        combined.insert(0, "family", family)
        all_assignments.append(combined)
        bands = _configured_bands(family, config)
        expected_dev_keys = set(zip(dev_rows["CUSTOMERNAME"].astype(str), dev_rows["PROJECTID"].astype(str)))
        expected_hold_keys = set(zip(hold_rows["CUSTOMERNAME"].astype(str), hold_rows["PROJECTID"].astype(str)))
        actual_dev_keys = set(zip(oof_predictions["CUSTOMERNAME"].astype(str), oof_predictions["PROJECTID"].astype(str)))
        actual_hold_keys = set(zip(holdout_predictions["CUSTOMERNAME"].astype(str), holdout_predictions["PROJECTID"].astype(str)))
        prediction_join_detail = {
            "cv_oof_missing_predictions": len(expected_dev_keys - actual_dev_keys),
            "holdout_missing_predictions": len(expected_hold_keys - actual_hold_keys),
            "cv_oof_duplicate_predictions": int(oof_predictions.duplicated(["CUSTOMERNAME", "PROJECTID"]).sum()),
            "holdout_duplicate_predictions": int(holdout_predictions.duplicated(["CUSTOMERNAME", "PROJECTID"]).sum()),
        }
        checks = [
            {"name": "prediction_join_complete",
             "status": "pass" if not any(prediction_join_detail.values()) else "fail",
             "detail": prediction_join_detail},
            {"name": "source_field_present",
             "status": "pass" if source_detail["status"] != "fail" or source_detail.get("reason") != "source_field_missing" else "fail",
             "detail": source_detail},
            {"name": "assignment_complete",
             "status": "pass" if not dev_band.isna().any() and not hold_band.isna().any() else "fail",
             "detail": {"development_missing": int(dev_band.isna().sum()), "holdout_missing": int(hold_band.isna().sum())},
            },
            {"name": "band_count_exact",
             "status": "pass" if set(dev_band.dropna().unique()) == set(bands) and set(hold_band.dropna().unique()) == set(bands) else "fail",
             "detail": {"expected": bands}},
        ]
        cv_support = _support(dev_rows, "cv_oof", family, bands,
                              config.development_minimum_rows_per_band,
                              config.development_minimum_rows_per_class_per_band)
        hold_support = _support(hold_rows, "locked_holdout", family, bands,
                                config.holdout_minimum_rows_per_band,
                                config.holdout_minimum_rows_per_class_per_band)
        for name, support in (("cv_total_support", cv_support), ("cv_class_support", cv_support),
                              ("holdout_total_support", hold_support), ("holdout_class_support", hold_support)):
            criterion = "minimum_rows" if "total" in name else "minimum_class_support"
            checks.append({"name": name,
                           "status": "pass" if all(criterion not in item["failed_criteria"] for item in support) else "fail",
                           "detail": support})
        eligible = all(item["status"] == "pass" for item in checks)
        metric_rows = []
        if eligible:
            for population, data, preds in (
                ("cv_oof", dev_rows, oof_predictions),
                ("locked_holdout", hold_rows, holdout_predictions),
            ):
                prediction_columns = [
                    "CUSTOMERNAME", "PROJECTID", "PREDICTED",
                    "PROBABILITY_NO_DELAY", "PROBABILITY_MILD_DELAY",
                    "PROBABILITY_SIGNIFICANT_DELAY",
                ]
                keyed = data.merge(
                    preds[prediction_columns],
                    on=["CUSTOMERNAME", "PROJECTID"],
                    how="left",
                    validate="one_to_one",
                )
                for band in bands:
                    subset = keyed[keyed["BAND"] == band]
                    y = subset[LABEL_COLUMN].to_numpy(dtype=int)
                    probability = subset[["PROBABILITY_NO_DELAY", "PROBABILITY_MILD_DELAY", "PROBABILITY_SIGNIFICANT_DELAY"]].to_numpy(dtype=float)
                    prediction = subset["PREDICTED"].to_numpy(dtype=int)
                    metric_rows.append({"population": population, "band": band,
                                        "metrics": classification_metrics(y, prediction, probability, weight, calibration_bins)})
        families[family] = {
            "family": family, "eligible": bool(eligible), "bands": bands,
            "source_details": [source_detail, hold_detail],
            "checks": checks, "support": cv_support + hold_support,
            "metrics": metric_rows,
        }
    assignments = pd.concat(all_assignments, ignore_index=True)
    return {"schema_version": "schedule-subgroups-v1", "families": families,
            "assignments": assignments}
