from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ALLOWED_BEGINNING = {
    "Yes",
    "Yes, if planned duration valid",
    "Yes, if planned end date exists",
    "Yes, if closure dates entered",
    "Yes, if closure date is entered at setup",
}
KEYWORD_PREFIXES = ("PROJ_KW_", "CONTRACT_KW_", "ITEM_KW_")
EXCLUDED = {
    "RECORD_ID", "PROJECTID", "PROJECTNAME", "PROJECTDESCRIPTION", "PROJECTCODE",
    "CUSTOMERNAME", "PROJECTSTATUS", "PERCENTDELAYED",
}


def select_features(frame, dictionary):
    beginning = dict(zip(dictionary["FieldName"], dictionary["BeginningAvailable"]))
    target_fields = set(dictionary.loc[
        dictionary["FieldGroup"].eq("target/post-payment"), "FieldName"
    ])
    target_fields |= {column for column in frame.columns if column.startswith("TARGET")}
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    keywords = [column for column in numeric if column.startswith(KEYWORD_PREFIXES)]
    nonkeywords = [
        column for column in numeric
        if beginning.get(column) in ALLOWED_BEGINNING
        and column not in EXCLUDED
        and column not in target_fields
    ]
    return sorted(set(keywords) | set(nonkeywords))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv")
    parser.add_argument("--field-dictionary", default="project_feature_non_keyword_field_dictionary_2026-06-10.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--n-estimators", type=int, default=250)
    args = parser.parse_args()

    frame = pd.read_csv(args.data, low_memory=False)
    dictionary = pd.read_csv(args.field_dictionary)
    frame = frame[frame["PERCENTDELAYED"].notna()].copy()
    features = select_features(frame, dictionary)
    labels = pd.cut(
        frame["PERCENTDELAYED"],
        [-np.inf, 0, 25, np.inf],
        labels=[0, 1, 2],
        right=True,
    ).astype(int)
    train_x, test_x, train_y, test_y = train_test_split(
        frame[features], labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=8,
            min_samples_leaf=15,
            min_samples_split=10,
            max_features=0.15,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(train_x, train_y)
    predictions = pipeline.predict(test_x)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "macro_f1": float(f1_score(test_y, predictions, average="macro")),
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "feature_count": len(features),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "schedule_risk_model.joblib", compress=3)
    schema = {
        "feature_schema_version": "schedule-project-features-v1",
        "keyword_manifest_version": "approved-keywords-2026-06-11",
        "ordered_features": features,
        "class_labels": ["no_delay", "mild_delay", "significant_delay"],
    }
    (output / "schedule_risk_feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    card = {
        "model_version": "schedule-rf-dev-2026-07-18.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_proof_of_concept",
        "metrics": metrics,
        "training_data": str(args.data),
        "target": "PercentDelayed bins <=0, (0,25], >25",
        "limitations": [
            "Development artifact pending client feature and target approval",
            "Must be retrained from the production-safe Snowflake feature calculation",
        ],
    }
    (output / "schedule_risk_model_card.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "schedule_risk_training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

