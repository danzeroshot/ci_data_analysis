from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .config import Settings
from .errors import FeatureRepositoryNotReady, ScheduleRiskError
from .feature_schema import FeatureSchema
from .feature_store import LocalFeatureRepository
from .model_runtime import ModelRuntime


settings = Settings.load()
if settings.model_bundle_path:
    runtime = ModelRuntime.from_bundle(settings.model_bundle_path)
    schema = runtime.schema
else:
    schema = FeatureSchema.load(settings.feature_schema_path)
    runtime = ModelRuntime(settings.model_path, settings.model_card_path, schema)
if settings.feature_repository != "local":
    raise NotImplementedError(
        "Snowflake repository activation requires deployment table configuration"
    )
repository = LocalFeatureRepository(
    settings.feature_snapshot_root, schema, settings.max_feature_age_hours
)


def _load_customer_runtimes():
    runtimes = {}
    root = settings.customer_model_root
    if not root.exists():
        return runtimes
    for customer_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        pointer = customer_dir / "current.json"
        if not pointer.exists():
            continue
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            relative = Path(data["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                continue
            bundle = (customer_dir / relative).resolve()
            if customer_dir not in bundle.parents:
                continue
            runtimes[customer_dir.name] = ModelRuntime.from_bundle(bundle)
        except Exception:
            continue
    return runtimes


customer_runtimes = _load_customer_runtimes()
try:
    repository.open()
except ScheduleRiskError:
    pass


def repository_ready() -> bool:
    try:
        ensure_repository()
        repository.fetch([])
        return True
    except Exception:
        return False


def ensure_repository() -> None:
    if repository._frame is None:
        repository.open()
    else:
        repository.refresh_if_changed()


def _safe_customer_key(value):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unknown"


def score_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(payload.get("request_id", ""))
    projects = payload.get("projects") or []
    if not request_id:
        raise ValueError("request_id is required")
    if not projects or len(projects) > settings.max_batch_size:
        raise ValueError(
            "projects must contain 1 to {} items".format(settings.max_batch_size)
        )
    ensure_repository()
    keys = [(item["customer_name"], str(item["project_id"])) for item in projects]
    frame, missing, metadata = repository.fetch(keys)
    predictions = runtime.predict(frame) if not frame.empty else []
    customer_predictions = {}
    for customer_key, customer_runtime in customer_runtimes.items():
        customer_predictions[customer_key] = customer_runtime.predict(frame) if not frame.empty else []
    by_key = {}
    for index, (unused, row) in enumerate(frame.iterrows()):
        key = (str(row["CUSTOMERNAME"]), str(row["PROJECTID"]))
        by_key[key] = {
            "all_customer": predictions[index],
            "customer_specific": (
                customer_predictions.get(_safe_customer_key(row["CUSTOMERNAME"]), [None] * len(frame))[index]
                if _safe_customer_key(row["CUSTOMERNAME"]) in customer_predictions else None
            ),
        }
    results = []
    for customer, project in keys:
        prediction = by_key.get((str(customer), str(project)))
        if prediction is None:
            results.append({
                "customer_name": customer,
                "project_id": project,
                "status": "not_scored",
                "error": {"code": "FEATURE_ROW_NOT_FOUND", "retryable": False},
            })
        else:
            customer_result = prediction["customer_specific"]
            results.append({
                "customer_name": customer,
                "project_id": project,
                "status": "scored",
                "prediction": prediction["all_customer"],
                "models": {
                    "all_customer": {
                        "status": "available",
                        "model_version": runtime.model_card["model_version"],
                        "prediction": prediction["all_customer"],
                    },
                    "customer_specific": (
                        {
                            "status": "available",
                            "model_version": customer_runtimes[_safe_customer_key(customer)].model_card["model_version"],
                            "prediction": customer_result,
                        }
                        if customer_result is not None else {
                            "status": "unavailable",
                            "reason_code": "customer_model_not_released",
                            "detail": "All-customer result remains valid",
                        }
                    ),
                },
                "feature_build_id": metadata["build_id"],
                "feature_age_hours": metadata["feature_age_hours"],
            })
    return {
        "request_id": request_id,
        "agent": "schedule-risk-agent",
        "model_version": runtime.model_card["model_version"],
        "feature_schema_version": schema.feature_schema_version,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": settings.feature_repository,
        "results": results,
    }


def main():
    from .mcp_http_sse import serve

    serve(score_payload, repository_ready)


if __name__ == "__main__":
    main()
