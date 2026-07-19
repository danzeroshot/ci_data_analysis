from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from .contracts import CLASS_LABELS


COLORS = ["#2878B5", "#E6A73C", "#C64242"]
DISPLAY_LABELS = {
    "no_delay": "No delay",
    "mild_delay": "Mild delay",
    "significant_delay": "Significant delay",
}


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, (np.integer, int)):
        return "{:,}".format(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "Not available"
        return ("{:." + str(digits) + "f}").format(float(value))
    return str(value)


def _format_percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "Not available"
    return ("{:." + str(digits) + "%}").format(float(value))


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:image/png;base64," + encoded


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]], classes: Optional[Sequence[str]] = None) -> str:
    header = "".join("<th>{}</th>".format(html.escape(str(value))) for value in headers)
    body_rows = []
    for index, row in enumerate(rows):
        row_class = ""
        if classes is not None:
            row_class = " class='{}'".format(html.escape(classes[index]))
        cells = "".join("<td>{}</td>".format(html.escape(str(value))) for value in row)
        body_rows.append("<tr{}>{}</tr>".format(row_class, cells))
    return (
        "<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(
            header, "".join(body_rows)
        )
    )


def _status_table(rows: List[Dict[str, str]]) -> str:
    return _table(
        ["Assessment area", "Status", "What was actually accomplished"],
        [
            (
                row["area"],
                row["status"].replace("_", " ").title(),
                row["detail"],
            )
            for row in rows
        ],
        [row["status"].replace("_", "-") for row in rows],
    )


def _save_confusion(metrics: Dict[str, Any], output: Path, normalized: bool, title_prefix: str = "Locked Holdout") -> None:
    key = "confusion_matrix_actual_normalized" if normalized else "confusion_matrix_count"
    matrix = np.asarray(metrics[key], dtype=float)
    fig, axis = plt.subplots(figsize=(6.6, 5.4))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    for row in range(3):
        for column in range(3):
            text = "{:.1%}".format(matrix[row, column]) if normalized else "{:,.0f}".format(
                matrix[row, column]
            )
            threshold = 0.55 if normalized else matrix.max() * 0.55
            axis.text(
                column, row, text, ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "#17202A",
            )
    labels = [DISPLAY_LABELS[label] for label in CLASS_LABELS]
    axis.set_xticks(range(3), labels, rotation=25, ha="right")
    axis.set_yticks(range(3), labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title(
        "{} Confusion Matrix ({})".format(title_prefix,
            "Row Normalized" if normalized else "Counts"
        )
    )
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_class_metrics(metrics: Dict[str, Any], output: Path) -> None:
    values = metrics["per_class"]
    labels = list(CLASS_LABELS)
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for offset, metric, color in [
        (-width, "precision", "#2878B5"),
        (0, "recall", "#C64242"),
        (width, "f1", "#4E9F62"),
    ]:
        axis.bar(
            x + offset,
            [values[label][metric] or 0 for label in labels],
            width,
            label=metric.title(),
            color=color,
        )
    axis.set_xticks(x, [DISPLAY_LABELS[label] for label in labels])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Locked Holdout Performance by Class")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_class_distribution(split: Dict[str, Any], output: Path) -> None:
    development = split.get("development_class_counts", {})
    holdout = split.get("holdout_class_counts", {})
    x = np.arange(3)
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    axis.bar(
        x - width / 2,
        [int(development.get(str(class_id), 0)) for class_id in range(3)],
        width,
        label="Development",
        color="#2878B5",
    )
    axis.bar(
        x + width / 2,
        [int(holdout.get(str(class_id), 0)) for class_id in range(3)],
        width,
        label="Locked holdout",
        color="#E6A73C",
    )
    axis.set_xticks(x, [DISPLAY_LABELS[label] for label in CLASS_LABELS])
    axis.set_ylabel("Projects")
    axis.set_title("Class Support by Evaluation Population")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_metric_comparison(reference: Dict[str, Any], output: Path) -> None:
    populations = [
        ("Training", reference["development_in_sample"]["overall"]),
        ("CV OOF", reference["cross_validation_out_of_fold"]["overall"]),
        ("Locked holdout", reference["locked_holdout"]["overall"]),
    ]
    metrics = [
        ("macro_f1", "Macro F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("significant_delay_recall", "Significant-delay recall"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    for index, (population, values) in enumerate(populations):
        axis.bar(
            x + (index - 1) * width,
            [values.get(name) or 0 for name, _ in metrics],
            width,
            label=population,
            color=["#7F8C8D", "#2878B5", "#4E9F62"][index],
        )
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Training, Cross-Validation, and Locked-Holdout Performance")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _prediction_arrays(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    actual = frame["SCHEDULERISKBIN"].to_numpy(dtype=int)
    probabilities = np.column_stack([
        frame["PROBABILITY_" + label.upper()].to_numpy(dtype=float)
        for label in CLASS_LABELS
    ])
    return actual, probabilities


def _save_roc_pr_curves(frame: pd.DataFrame, roc_output: Path, pr_output: Path) -> None:
    actual, probabilities = _prediction_arrays(frame)
    fig_roc, axis_roc = plt.subplots(figsize=(6.8, 5.2))
    fig_pr, axis_pr = plt.subplots(figsize=(6.8, 5.2))
    for class_id, (label, color) in enumerate(zip(CLASS_LABELS, COLORS)):
        binary = (actual == class_id).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(binary, probabilities[:, class_id])
        precision, recall, _ = precision_recall_curve(binary, probabilities[:, class_id])
        axis_roc.plot(
            false_positive,
            true_positive,
            color=color,
            label="{} (AUC {:.3f})".format(
                DISPLAY_LABELS[label], auc(false_positive, true_positive)
            ),
        )
        axis_pr.plot(
            recall,
            precision,
            color=color,
            label="{} (PR AUC {:.3f})".format(
                DISPLAY_LABELS[label], np.trapz(precision[::-1], recall[::-1])
            ),
        )
    axis_roc.plot([0, 1], [0, 1], color="#7F8C8D", linestyle="--")
    axis_roc.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Locked Holdout One-vs-Rest ROC",
    )
    axis_pr.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Locked Holdout Precision-Recall",
    )
    for axis in (axis_roc, axis_pr):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(roc_output, dpi=150)
    fig_pr.savefig(pr_output, dpi=150)
    plt.close(fig_roc)
    plt.close(fig_pr)


def _save_calibration_curves(frame: pd.DataFrame, output: Path) -> None:
    actual, probabilities = _prediction_arrays(frame)
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    edges = np.linspace(0.0, 1.0, 11)
    for class_id, (label, color) in enumerate(zip(CLASS_LABELS, COLORS)):
        binary = (actual == class_id).astype(int)
        observed = []
        predicted = []
        for index in range(len(edges) - 1):
            upper = probabilities[:, class_id] <= edges[index + 1] if index == 9 else (
                probabilities[:, class_id] < edges[index + 1]
            )
            mask = (probabilities[:, class_id] >= edges[index]) & upper
            if mask.any():
                predicted.append(float(probabilities[mask, class_id].mean()))
                observed.append(float(binary[mask].mean()))
        axis.plot(predicted, observed, marker="o", color=color, label=DISPLAY_LABELS[label])
    axis.plot([0, 1], [0, 1], color="#7F8C8D", linestyle="--", label="Ideal")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed class frequency",
        title="Locked Holdout Calibration",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_candidates(candidate_rows: List[Dict[str, Any]], output: Path) -> None:
    valid = [row for row in candidate_rows if row.get("status") == "succeeded"]
    valid.sort(key=lambda row: row.get("rank", 10 ** 9))
    valid = valid[:20]
    fig, axis = plt.subplots(figsize=(8.4, max(3.5, len(valid) * 0.42)))
    axis.barh(
        np.arange(len(valid)),
        [row["aggregate"]["overall"]["selection_score"] for row in valid],
        color="#2878B5",
    )
    axis.set_yticks(np.arange(len(valid)), [row["candidate_id"] for row in valid])
    axis.invert_yaxis()
    axis.set_xlabel("Cross-validation selection score")
    axis.set_title("Completed Random-Forest Candidate Ranking")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_customer_metrics(customer_metrics: Dict[str, Any], output: Path) -> bool:
    eligible = [
        item for item in customer_metrics.get("customers", [])
        if item.get("metrics")
    ]
    if not eligible:
        return False
    names = [item["customer"] for item in eligible]
    macro = [item["metrics"]["overall"]["macro_f1"] for item in eligible]
    significant = [
        item["metrics"]["overall"]["significant_delay_recall"] for item in eligible
    ]
    x = np.arange(len(names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.bar(x - width / 2, macro, width, label="Macro F1", color="#2878B5")
    axis.bar(
        x + width / 2,
        significant,
        width,
        label="Significant-delay recall",
        color="#C64242",
    )
    axis.set_xticks(x, names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Leave-One-Customer-Out Generalization")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def _save_feature_importance(frame: pd.DataFrame, output: Path) -> bool:
    if frame.empty:
        return False
    top = frame.head(20).sort_values("importance")
    fig, axis = plt.subplots(figsize=(9.2, 7.0))
    axis.barh(np.arange(len(top)), top["importance"], color="#2878B5")
    axis.set_yticks(np.arange(len(top)), top["feature_name"], fontsize=8)
    axis.set_xlabel("Mean decrease in impurity")
    axis.set_title("Top 20 Random-Forest Feature Importances")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return True


def _metric_comparison_table(reference: Dict[str, Any], confidence: Dict[str, Any]) -> str:
    metric_rows = []
    names = [
        ("macro_f1", "Macro F1"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("significant_delay_recall", "Significant-delay recall"),
        ("roc_auc_ovr_macro", "OVR macro ROC AUC"),
        ("log_loss", "Log loss"),
        ("expected_calibration_error", "Expected calibration error"),
    ]
    intervals = confidence.get("metrics", {})
    for name, label in names:
        interval = intervals.get(name, {})
        ci = (
            "{} to {}".format(
                _format_number(interval.get("lower_95")),
                _format_number(interval.get("upper_95")),
            )
            if interval else "Not calculated"
        )
        metric_rows.append((
            label,
            _format_number(reference["development_in_sample"]["overall"].get(name)),
            _format_number(reference["cross_validation_out_of_fold"]["overall"].get(name)),
            _format_number(reference["locked_holdout"]["overall"].get(name)),
            ci,
        ))
    return _table(
        ["Metric", "Training", "CV out-of-fold", "Locked holdout", "Holdout 95% CI"],
        metric_rows,
    )


def _build_assessment(
    run_summary: Dict[str, Any],
    temporal: Dict[str, Any],
    customers: Dict[str, Any],
    importance_available: bool,
) -> List[Dict[str, str]]:
    execution = run_summary.get("execution_assessment", {})
    temporal_status = execution.get("temporal_evaluation", {}).get("status")
    temporal_reason = execution.get("temporal_evaluation", {}).get("reason")
    if temporal_status is None:
        temporal_status = "completed" if temporal.get("available") else "unavailable"
        temporal_reason = temporal.get("reason")
    customer_execution = execution.get("customer_isolation", {})
    customer_status = customer_execution.get("status")
    if customer_status == "disabled":
        customer_status = "not_run"
    if customer_status is None:
        customer_status = (
            "completed" if customers.get("customers") else "not_run"
        )
    eligible_customers = customer_execution.get(
        "eligible_customers",
        sum(bool(item.get("metrics")) for item in customers.get("customers", [])),
    )
    labeled_customers = customer_execution.get(
        "labeled_customers", len(customers.get("customers", []))
    )
    feature_customers = customer_execution.get(
        "feature_snapshot_customers", labeled_customers
    )
    missing_label_customers = customer_execution.get(
        "missing_label_customers", []
    )
    configured = execution.get("configured_candidate_count")
    completed = execution.get("completed_candidate_count")
    run_scope = execution.get("run_scope", "unknown")
    search_status = "limited" if run_scope == "benchmark_search" else "completed"
    return [
        {
            "area": "Immutable input reconciliation",
            "status": "completed",
            "detail": "Feature and label snapshots were verified and joined with explicit unmatched counts.",
        },
        {
            "area": "Feature qualification",
            "status": "completed",
            "detail": "Beginning-only candidates were screened on development rows before fitting.",
        },
        {
            "area": "Random-forest search",
            "status": search_status,
            "detail": "{} of {} configured candidates completed; run scope is {}.".format(
                completed if completed is not None else "Unknown",
                configured if configured is not None else "Unknown",
                run_scope,
            ),
        },
        {
            "area": "Locked-holdout evaluation",
            "status": "completed",
            "detail": "The serialized release candidate was evaluated on the untouched hash holdout.",
        },
        {
            "area": "Temporal evaluation",
            "status": temporal_status or "unavailable",
            "detail": (
                "Completed on the configured time split."
                if temporal.get("available")
                else (
                    "Disabled by run configuration."
                    if temporal_status == "not_run"
                    else "Unavailable: {}.".format(temporal_reason or "no temporal result")
                )
            ),
        },
        {
            "area": "Customer-isolation evaluation",
            "status": customer_status,
            "detail": (
                "Disabled by run configuration."
                if customer_status == "not_run"
                else (
                    "{} of {} feature-snapshot customers were evaluated; {} had matched labels. "
                    "Customers without matched labels: {}."
                ).format(eligible_customers, feature_customers, labeled_customers,
                         ", ".join(missing_label_customers) or "none")
            ),
        },
        {
            "area": "Feature importance",
            "status": "completed" if importance_available else "unavailable",
            "detail": (
                "Impurity-based importance was exported from the fitted forest."
                if importance_available else "The run did not export model importance."
            ),
        },
        {
            "area": "External drift evaluation",
            "status": "not_run",
            "detail": "No later external feature/label snapshot was supplied to this training run.",
        },
        {
            "area": "Planned subgroup evaluations",
            "status": (
                "completed"
                if run_summary.get("subgroup_eligibility")
                and all(run_summary["subgroup_eligibility"].values())
                else "unavailable"
            ),
            "detail": (
                "All required subgroup families passed data eligibility; subgroup performance is diagnostic."
                if run_summary.get("subgroup_eligibility")
                and all(run_summary["subgroup_eligibility"].values())
                else "Subgroup eligibility was not available or at least one required family failed."
            ),
        },
        {
            "area": "Customer-specific models",
            "status": (
                "completed" if run_summary.get("customer_models") else "not_run"
            ),
            "detail": (
                "Customer tuning, global-parameter fallback, and unavailable dispositions were evaluated."
                if run_summary.get("customer_models")
                else "Customer-specific model training was disabled by configuration."
            ),
        },
        {
            "area": "Incumbent comparison",
            "status": "not_run",
            "detail": "No incumbent bundle was configured as a paired comparison baseline.",
        },
        {
            "area": "Production release validation",
            "status": "blocked",
            "detail": "Client approvals, numeric release thresholds, production label lineage, and Docker validation are incomplete.",
        },
    ]


def generate_report(
    run_dir: Path,
    run_summary: Dict[str, Any],
    holdout_metrics: Dict[str, Any],
    candidate_results: List[Dict[str, Any]],
    release_gates: List[Dict[str, Any]],
) -> Path:
    run_dir = Path(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    reconciliation = _read_json(run_dir / "reconciliation.json", {})
    split = _read_json(run_dir / "split_summary.json", {})
    qualification = _read_json(run_dir / "qualification_summary.json", {})
    selected = _read_json(run_dir / "selected_hyperparameters.json", {})
    reference = _read_json(run_dir / "reference_metrics.json", {})
    confidence = _read_json(run_dir / "confidence_intervals.json", {})
    temporal = _read_json(run_dir / "temporal_metrics.json", {})
    customers = _read_json(run_dir / "customer_holdout_metrics.json", {})
    subgroup_eligibility = _read_json(
        run_dir / "subgroups" / "subgroup_eligibility.json", {}
    )
    customer_models = _read_json(
        run_dir / "customer_models" / "customer_model_eligibility.json", {}
    )
    feature_manifest = _read_json(run_dir / "feature_snapshot_manifest.json", {})
    label_manifest = _read_json(run_dir / "label_snapshot_manifest.json", {})
    environment = _read_json(run_dir / "environment.json", {})
    importance_path = run_dir / "feature_importances.parquet"
    importance = (
        pd.read_parquet(importance_path)
        if importance_path.is_file() else pd.DataFrame()
    )
    predictions = pd.read_parquet(run_dir / "locked_holdout_predictions.parquet")

    plot_files = {
        "distribution": plot_dir / "class_distribution.png",
        "metric_comparison": plot_dir / "train_cv_holdout.png",
        "confusion_count": plot_dir / "locked_holdout_confusion_count.png",
        "confusion_normalized": plot_dir / "locked_holdout_confusion_normalized.png",
        "class_metrics": plot_dir / "locked_holdout_class_metrics.png",
        "roc": plot_dir / "locked_holdout_roc.png",
        "pr": plot_dir / "locked_holdout_pr.png",
        "calibration": plot_dir / "locked_holdout_calibration.png",
        "candidates": plot_dir / "candidate_ranking.png",
        "customers": plot_dir / "customer_isolation.png",
        "importance": plot_dir / "feature_importance.png",
        "temporal": plot_dir / "temporal_confusion_normalized.png",
    }
    _save_class_distribution(split, plot_files["distribution"])
    _save_metric_comparison(reference, plot_files["metric_comparison"])
    _save_confusion(holdout_metrics, plot_files["confusion_count"], normalized=False)
    _save_confusion(holdout_metrics, plot_files["confusion_normalized"], normalized=True)
    _save_class_metrics(holdout_metrics, plot_files["class_metrics"])
    _save_roc_pr_curves(predictions, plot_files["roc"], plot_files["pr"])
    _save_calibration_curves(predictions, plot_files["calibration"])
    _save_candidates(candidate_results, plot_files["candidates"])
    customer_plot = _save_customer_metrics(customers, plot_files["customers"])
    importance_plot = _save_feature_importance(importance, plot_files["importance"])
    temporal_plot = bool(temporal.get("available") and temporal.get("metrics"))
    if temporal_plot:
        _save_confusion(
            temporal["metrics"], plot_files["temporal"], normalized=True, title_prefix="Temporal Holdout"
        )

    assessment = _build_assessment(
        run_summary, temporal, customers, not importance.empty
    )
    overall = holdout_metrics["overall"]
    cards = [
        ("Run type", run_summary.get("execution_assessment", {}).get("run_scope", "Unknown")),
        ("Matched projects", _format_number(reconciliation.get("matched_rows"))),
        ("Locked holdout", _format_number(split.get("locked_holdout_rows"))),
        ("Macro F1", _format_number(overall.get("macro_f1"))),
        ("Balanced accuracy", _format_number(overall.get("balanced_accuracy"))),
        ("Significant-delay recall", _format_number(overall.get("significant_delay_recall"))),
    ]
    cards_html = "".join(
        "<div class='metric'><span>{}</span><strong>{}</strong></div>".format(
            html.escape(str(label)), html.escape(str(value))
        )
        for label, value in cards
    )

    data_rows = [
        ("Feature snapshot rows", _format_number(reconciliation.get("feature_rows"))),
        ("Valid labeled rows", _format_number(reconciliation.get("label_rows"))),
        ("Matched model rows", _format_number(reconciliation.get("matched_rows"))),
        ("Feature-only rows", _format_number(reconciliation.get("feature_only_rows"))),
        ("Feature-to-label match rate", _format_percent(reconciliation.get("match_rate_from_features"))),
        ("Development rows", _format_number(split.get("development_rows"))),
        ("Locked-holdout rows", _format_number(split.get("locked_holdout_rows"))),
        ("Label source", label_manifest.get("source_type", "Unknown")),
        ("Development-only labels", str(label_manifest.get("development_only", "Unknown"))),
    ]
    class_rows = []
    total_counts = reconciliation.get("class_counts", {})
    development_counts = split.get("development_class_counts", {})
    holdout_counts = split.get("holdout_class_counts", {})
    for class_id, label in enumerate(CLASS_LABELS):
        class_rows.append((
            DISPLAY_LABELS[label],
            _format_number(total_counts.get(str(class_id))),
            _format_number(development_counts.get(str(class_id))),
            _format_number(holdout_counts.get(str(class_id))),
        ))

    per_class_rows = []
    for label in CLASS_LABELS:
        values = holdout_metrics["per_class"][label]
        per_class_rows.append((
            DISPLAY_LABELS[label],
            _format_number(values.get("support")),
            _format_percent(values.get("prevalence")),
            _format_number(values.get("precision")),
            _format_number(values.get("recall")),
            _format_number(values.get("f1")),
            _format_number(values.get("false_negative")),
            _format_percent(values.get("false_negative_rate")),
            _format_number(values.get("roc_auc_ovr")),
            _format_number(values.get("average_precision")),
        ))

    customer_rows = []
    for item in customers.get("customers", []):
        values = item.get("metrics", {}).get("overall", {})
        customer_rows.append((
            item.get("customer"),
            "Evaluated" if item.get("metrics") else "Unavailable",
            _format_number(item.get("test_rows")),
            json.dumps(item.get("class_counts", {}), sort_keys=True),
            _format_number(values.get("macro_f1")),
            _format_number(values.get("balanced_accuracy")),
            _format_number(values.get("significant_delay_recall")),
            item.get("reason") or "",
        ))

    reported_customers = {str(row[0]) for row in customer_rows}
    for customer in sorted(
        set(feature_manifest.get("customers", [])) - reported_customers
    ):
        customer_rows.append((
            customer,
            "Unavailable",
            "0",
            "{}",
            "Not available", "Not available", "Not available",
            "no_matched_labels",
        ))
    customer_rows.sort(key=lambda row: str(row[0]))
    subgroup_rows = []
    for family, item in sorted(subgroup_eligibility.get("families", {}).items()):
        failed = [
            check["name"] for check in item.get("checks", [])
            if check.get("status") == "fail"
        ]
        subgroup_rows.append((
            family,
            "Eligible" if item.get("eligible") else "Ineligible",
            ", ".join(item.get("bands", [])),
            ", ".join(failed) or "None",
        ))
    if not subgroup_rows:
        subgroup_rows = [("All families", "Unavailable", "Not calculated", "subgroup evaluation disabled")]

    customer_model_rows = []
    for item in customer_models.get("customers", []):
        customer_model_rows.append((
            item.get("customer"),
            item.get("status", "Unavailable").replace("_", " ").title(),
            item.get("training_mode") or "None",
            item.get("reason_code") or "None",
        ))
    if not customer_model_rows:
        customer_model_rows = [("All customers", "Not run", "None", "customer modeling disabled")]

    qualification_rows = [
        ("Candidate fields", _format_number(qualification.get("candidate_count"))),
        ("Qualified fields", _format_number(qualification.get("accepted_count"))),
        ("Rejected fields", _format_number(qualification.get("rejected_count"))),
    ] + [
        ("Rejected: " + reason.replace("_", " "), _format_number(count))
        for reason, count in sorted(qualification.get("rejection_counts", {}).items())
    ]

    candidate_rows = []
    for item in sorted(
        [row for row in candidate_results if row.get("status") == "succeeded"],
        key=lambda row: row.get("rank", 10 ** 9),
    ):
        aggregate = item["aggregate"]
        candidate_rows.append((
            item.get("rank"),
            item["candidate_id"],
            _format_number(aggregate["overall"].get("selection_score")),
            _format_number(aggregate["overall"].get("macro_f1")),
            _format_number(aggregate["overall"].get("significant_delay_recall")),
            _format_number(aggregate.get("train_to_validation_macro_f1_gap")),
            _format_number(item.get("elapsed_seconds"), 1),
        ))

    importance_rows = [
        (
            int(row.rank),
            row.feature_name,
            row.family,
            _format_number(row.importance, 6),
        )
        for row in importance.head(20).itertuples(index=False)
    ]

    gate_rows = [
        (
            gate["name"],
            gate["status"].title(),
            gate.get("detail", ""),
        )
        for gate in release_gates
    ]
    gate_classes = [gate["status"].lower() for gate in release_gates]

    temporal_state = run_summary.get("execution_assessment", {}).get(
        "temporal_evaluation", {}
    ).get("status")
    if temporal.get("available"):
        temporal_status = "Completed"
        temporal_detail = "Temporal metrics were calculated."
    elif temporal_state == "not_run":
        temporal_status = "Not Run"
        temporal_detail = (
            "The temporal evaluation was disabled by this run's configuration; "
            "no result was estimated or substituted."
        )
    else:
        temporal_status = "Unavailable"
        temporal_detail = (
            "Reason: {}. This result was not estimated or silently substituted."
            .format(temporal.get("reason", "unknown"))
        )

    temporal_overall = temporal.get("metrics", {}).get("overall", {})
    temporal_rows = []
    if temporal.get("available"):
        temporal_rows = [
            ("Boundary date", temporal.get("boundary_date")),
            ("Training projects", _format_number(temporal.get("train_rows"))),
            ("Test projects", _format_number(temporal.get("test_rows"))),
            ("Excluded missing dates", _format_number(temporal.get("excluded_missing_date_count"))),
            ("Test class counts", json.dumps(temporal.get("test_class_counts", {}), sort_keys=True)),
            ("Macro F1", _format_number(temporal_overall.get("macro_f1"))),
            ("Balanced accuracy", _format_number(temporal_overall.get("balanced_accuracy"))),
            ("Significant-delay recall", _format_number(temporal_overall.get("significant_delay_recall"))),
            ("OVR macro ROC AUC", _format_number(temporal_overall.get("roc_auc_ovr_macro"))),
            ("Log loss", _format_number(temporal_overall.get("log_loss"))),
            ("Expected calibration error", _format_number(temporal_overall.get("expected_calibration_error"))),
        ]

    lineage_rows = [
        ("Run ID", run_summary.get("run_id")),
        ("Model version", run_summary.get("model_version")),
        ("Model status", run_summary.get("status")),
        ("Feature build", feature_manifest.get("build_id")),
        ("Feature schema", feature_manifest.get("feature_schema_version")),
        ("Keyword manifest", feature_manifest.get("keyword_manifest_version")),
        ("Label build", label_manifest.get("build_id")),
        ("Target definition", label_manifest.get("target_definition_version")),
        ("Selection weight", selected.get("significant_delay_weight")),
        ("Selected candidate", selected.get("candidate_id")),
        ("Python", environment.get("python")),
        ("Docker validation", run_summary.get("docker_validation")),
    ]

    figures = {
        name: _image_data_uri(path)
        for name, path in plot_files.items()
        if path.is_file()
    }
    customer_figure = (
        "<img src='{}' alt='Customer isolation performance'>".format(figures["customers"])
        if customer_plot else "<p class='unavailable'>No customer-isolation chart was produced.</p>"
    )
    importance_figure = (
        "<img src='{}' alt='Feature importance'>".format(figures["importance"])
        if importance_plot else "<p class='unavailable'>Feature importance was not exported for this run.</p>"
    )
    importance_table = (
        _table(["Rank", "Feature", "Family", "Importance"], importance_rows)
        if importance_rows else "<p class='unavailable'>No feature-importance artifact exists.</p>"
    )

    temporal_table = (
        _table(["Temporal measure", "Value"], temporal_rows)
        if temporal_rows else "<p class='unavailable'>No temporal metrics were produced.</p>"
    )
    temporal_figure = (
        "<img src='{}' alt='Temporal normalized confusion matrix'>".format(figures["temporal"])
        if temporal_plot else ""
    )

    template = """<!doctype html>
<html><head><meta charset="utf-8"><title>Schedule Risk Training Assessment</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; color: #17202A; background: #F4F6F7; line-height: 1.45; }
header { background: #17324D; color: white; padding: 30px 6%; }
header p { max-width: 900px; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h2 { margin-top: 0; font-size: 22px; }
h3 { font-size: 17px; margin-top: 22px; }
.metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.metric { background: white; border-left: 4px solid #2878B5; padding: 15px; min-width: 0; }
.metric span { display: block; color: #566573; font-size: 13px; }
.metric strong { display: block; margin-top: 6px; font-size: 21px; overflow-wrap: anywhere; }
.panel { background: white; margin-top: 18px; padding: 22px; border: 1px solid #DDE3E7; }
.callout { border-left: 5px solid #B42318; background: #FFF5F3; padding: 16px; margin-top: 18px; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: start; }
img { width: 100%; height: auto; display: block; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; vertical-align: top; border-bottom: 1px solid #E5E8E8; padding: 9px; }
th { background: #F7F9FA; }
.completed td:nth-child(2), .pass td:nth-child(2) { color: #216E39; font-weight: bold; }
.limited td:nth-child(2), .warn td:nth-child(2) { color: #9A6700; font-weight: bold; }
.unavailable td:nth-child(2), .not-run td:nth-child(2) { color: #6C5B00; font-weight: bold; }
.blocked td:nth-child(2), .fail td:nth-child(2) { color: #B42318; font-weight: bold; }
.unavailable { color: #6C5B00; }
.note { color: #566573; font-size: 13px; }
code { overflow-wrap: anywhere; }
@media (max-width: 780px) { .metrics, .charts { grid-template-columns: 1fr; } main { padding: 14px; } }
</style></head><body>
<header>
<h1>Schedule Risk Model Training Assessment</h1>
<p>Accurate accounting of completed evaluation, limited work, unavailable results, and production blockers.</p>
</header>
<main>
<section class="callout">
<strong>Development assessment, not production validation.</strong>
This run used development-only historical CSV labels, its explicitly configured candidate search, unapproved release thresholds, and no Docker validation.
</section>
<section class="metrics">__CARDS__</section>

<section class="panel" id="execution-coverage">
<h2>Execution Coverage</h2>
<p>The status column distinguishes work that completed from work that was limited, unavailable, not run, or blocked.</p>
<div class="table-wrap">__ASSESSMENT__</div>
</section>

<section class="panel" id="data-coverage">
<h2>Data Coverage and Label Availability</h2>
<div class="charts">
<div><div class="table-wrap">__DATA_TABLE__</div></div>
<div><img src="__DISTRIBUTION_IMAGE__" alt="Class support"></div>
</div>
<h3>Class Counts</h3>
<div class="table-wrap">__CLASS_TABLE__</div>
<p class="note">The mild-delay class is substantially smaller than the other classes. Per-class metrics must be read with support counts.</p>
</section>

<section class="panel" id="model-selection">
<h2>Model Selection Scope</h2>
<p>This report shows every completed configured candidate. Candidate counts and execution mode are explicit; completing the configured search does not imply exhaustive optimization.</p>
<div class="charts">
<div class="table-wrap">__CANDIDATE_TABLE__</div>
<div><img src="__CANDIDATE_IMAGE__" alt="Candidate ranking"></div>
</div>
<h3>Selected Hyperparameters</h3>
<div class="table-wrap">__PARAMETER_TABLE__</div>
</section>

<section class="panel" id="generalization">
<h2>Training, Cross-Validation, and Holdout</h2>
<p>Training metrics are in-sample and optimistic. CV out-of-fold and locked-holdout results are the primary generalization evidence.</p>
<div class="charts">
<div class="table-wrap">__METRIC_TABLE__</div>
<div><img src="__METRIC_IMAGE__" alt="Training CV holdout comparison"></div>
</div>
</section>

<section class="panel" id="class-performance">
<h2>Locked-Holdout Class Performance</h2>
<div class="table-wrap">__PER_CLASS_TABLE__</div>
<div class="charts">
<div><img src="__CLASS_METRIC_IMAGE__" alt="Class performance"></div>
<div><img src="__CONFUSION_COUNT_IMAGE__" alt="Confusion matrix counts"></div>
</div>
<img src="__CONFUSION_NORMALIZED_IMAGE__" alt="Normalized confusion matrix">
</section>

<section class="panel" id="discrimination-calibration">
<h2>Discrimination and Calibration</h2>
<div class="charts">
<div><img src="__ROC_IMAGE__" alt="ROC curves"></div>
<div><img src="__PR_IMAGE__" alt="Precision recall curves"></div>
</div>
<img src="__CALIBRATION_IMAGE__" alt="Calibration curves">
</section>

<section class="panel" id="subgroup-evaluation">
<h2>Required Subgroup-Family Eligibility</h2>
<p>Each required family must pass assignment, support, class-support, and prediction-join checks before a new global model can be promoted. Performance metrics are shown only for eligible families and remain diagnostic until numeric subgroup thresholds are approved.</p>
<div class="table-wrap">__SUBGROUP_TABLE__</div>
</section>

<section class="panel" id="customer-models">
<h2>Customer-Specific Model Dispositions</h2>
<p>The all-customer model is mandatory. Customer-specific models are independently tuned when support permits, use all-customer hyperparameters when only the absolute floor passes, and are explicitly unavailable below that floor.</p>
<div class="table-wrap">__CUSTOMER_MODEL_TABLE__</div>
</section>

<section class="panel" id="customer-isolation">
<h2>Customer-Isolation Evaluation</h2>
<p>Each evaluated customer was scored by a separate model trained on all other customers.</p>
<div class="table-wrap">__CUSTOMER_TABLE__</div>
__CUSTOMER_FIGURE__
</section>

<section class="panel" id="temporal-evaluation">
<h2>Temporal Evaluation</h2>
<p><strong>__TEMPORAL_STATUS__.</strong> __TEMPORAL_DETAIL__</p>
<div class="charts">
<div class="table-wrap">__TEMPORAL_TABLE__</div>
<div>__TEMPORAL_FIGURE__</div>
</div>
</section>

<section class="panel" id="feature-qualification">
<h2>Feature Qualification</h2>
<div class="table-wrap">__QUALIFICATION_TABLE__</div>
<h3>Top Model Importances</h3>
<p class="note">These are random-forest mean-decrease-in-impurity importances. They are descriptive and can favor continuous or high-cardinality features.</p>
__IMPORTANCE_FIGURE__
<div class="table-wrap">__IMPORTANCE_TABLE__</div>
</section>

<section class="panel" id="release-gates">
<h2>Release Gates</h2>
<div class="table-wrap">__GATE_TABLE__</div>
</section>

<section class="panel" id="lineage">
<h2>Lineage and Reproducibility</h2>
<div class="table-wrap">__LINEAGE_TABLE__</div>
</section>
</main></body></html>"""

    replacements = {
        "__CARDS__": cards_html,
        "__ASSESSMENT__": _status_table(assessment),
        "__DATA_TABLE__": _table(["Measure", "Value"], data_rows),
        "__DISTRIBUTION_IMAGE__": figures["distribution"],
        "__CLASS_TABLE__": _table(
            ["Class", "All matched", "Development", "Locked holdout"], class_rows
        ),
        "__CANDIDATE_TABLE__": _table(
            ["Rank", "Candidate", "Selection", "Macro F1", "Significant recall", "Train-CV gap", "Seconds"],
            candidate_rows,
        ),
        "__CANDIDATE_IMAGE__": figures["candidates"],
        "__PARAMETER_TABLE__": _table(
            ["Parameter", "Value"],
            [(name, value) for name, value in sorted(selected.get("parameters", {}).items())],
        ),
        "__METRIC_TABLE__": _metric_comparison_table(reference, confidence),
        "__METRIC_IMAGE__": figures["metric_comparison"],
        "__SUBGROUP_TABLE__": _table(
            ["Family", "Eligibility", "Configured bands", "Failed checks"],
            subgroup_rows,
        ),
        "__CUSTOMER_MODEL_TABLE__": _table(
            ["Customer", "Disposition", "Training mode", "Reason"],
            customer_model_rows,
        ),
        "__PER_CLASS_TABLE__": _table(
            ["Class", "Support", "Prevalence", "Precision", "Recall", "F1", "False negatives", "FNR", "ROC AUC", "Average precision"],
            per_class_rows,
        ),
        "__CLASS_METRIC_IMAGE__": figures["class_metrics"],
        "__CONFUSION_COUNT_IMAGE__": figures["confusion_count"],
        "__CONFUSION_NORMALIZED_IMAGE__": figures["confusion_normalized"],
        "__ROC_IMAGE__": figures["roc"],
        "__PR_IMAGE__": figures["pr"],
        "__CALIBRATION_IMAGE__": figures["calibration"],
        "__CUSTOMER_TABLE__": _table(
            ["Customer", "Status", "Rows", "Class counts", "Macro F1", "Balanced accuracy", "Significant recall", "Reason"],
            customer_rows,
        ),
        "__TEMPORAL_TABLE__": temporal_table,
        "__TEMPORAL_FIGURE__": temporal_figure,
        "__CUSTOMER_FIGURE__": customer_figure,
        "__TEMPORAL_STATUS__": temporal_status,
        "__TEMPORAL_DETAIL__": temporal_detail,
        "__QUALIFICATION_TABLE__": _table(["Disposition", "Count"], qualification_rows),
        "__IMPORTANCE_FIGURE__": importance_figure,
        "__IMPORTANCE_TABLE__": importance_table,
        "__GATE_TABLE__": _table(
            ["Gate", "Status", "Detail"], gate_rows, gate_classes
        ),
        "__LINEAGE_TABLE__": _table(["Field", "Value"], lineage_rows),
    }
    document = template
    for placeholder, value in replacements.items():
        document = document.replace(placeholder, value)
    output = run_dir / "report.html"
    output.write_text(document, encoding="utf-8")
    return output
