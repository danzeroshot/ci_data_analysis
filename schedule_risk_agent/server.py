from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from .config import Settings
from .errors import FeatureRepositoryNotReady, ScheduleRiskError
from .feature_schema import FeatureSchema
from .feature_store import LocalFeatureRepository
from .model_runtime import ModelRuntime


settings = Settings.load()
schema = FeatureSchema.load(settings.feature_schema_path)
if settings.feature_repository != "local":
    raise NotImplementedError(
        "Snowflake repository activation requires deployment table configuration"
    )
repository = LocalFeatureRepository(
    settings.feature_snapshot_root, schema, settings.max_feature_age_hours
)
runtime = ModelRuntime(settings.model_path, settings.model_card_path, schema)
try:
    repository.open()
except ScheduleRiskError:
    pass


def repository_ready() -> bool:
    return repository._frame is not None


def ensure_repository() -> None:
    if repository._frame is None:
        repository.open()
    else:
        repository.refresh_if_changed()


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
    by_key = {}
    for (_, row), prediction in zip(frame.iterrows(), predictions):
        by_key[(str(row["CUSTOMERNAME"]), str(row["PROJECTID"]))] = prediction
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
            results.append({
                "customer_name": customer,
                "project_id": project,
                "status": "scored",
                "prediction": prediction,
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

