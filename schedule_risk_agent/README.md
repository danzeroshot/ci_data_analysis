# Schedule Risk Agent

## Host Development

Run all tests:

    python3 -m pytest -q

Build and publish a local feature snapshot from Snowflake:

    python3 -m schedule_risk_agent.feature_refresh --target local

Future feature snapshots retain PLANNEDSTARTDATE as split-only metadata. It is
used for temporal evaluation and is never part of the predictor matrix.

## Training Data

Generate the approved candidate manifest:

    python3 -m schedule_risk_agent.training_pipeline generate-manifest \
        --schema models/schedule_risk_feature_schema.json \
        --output config/schedule_candidate_features.json

Create a development-only label snapshot from the historical CSV:

    python3 -m schedule_risk_agent.training_pipeline snapshot-labels \
        --source-csv data_analysis/custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv \
        --output-root training_snapshots/labels \
        --target-version schedule-delay-v1

Create a production-eligible label snapshot directly from Snowflake:

    python3 -m schedule_risk_agent.training_pipeline snapshot-labels-snowflake \
        --sql schedule_risk_label_calculation.sql \
        --output-root training_snapshots/labels \
        --target-version schedule-delay-v1

CSV-derived labels are permanently marked development_only.

## Training Pipeline

Run qualification only:

    python3 -m schedule_risk_agent.training_pipeline qualify --config <run.json>

Run qualification and tuning:

    python3 -m schedule_risk_agent.training_pipeline tune --config <run.json>

Run the complete host pipeline:

    python3 -m schedule_risk_agent.training_pipeline run --config <run.json>

Promote a verified candidate to the local release pointer:

    python3 -m schedule_risk_agent.training_pipeline release \
        --run-dir <run-directory> \
        --output-root model_artifacts/schedule-risk

Production promotion additionally requires --production and every production
release gate to pass.

Compare a release against a later immutable dataset:

    python3 -m schedule_risk_agent.training_pipeline compare \
        --bundle <release-directory> \
        --features <features.parquet> \
        --feature-manifest <feature-manifest.json> \
        --labels <labels.parquet> \
        --label-manifest <label-manifest.json> \
        --output <evaluation-directory>

Rollback changes only the atomic current.json pointer:

    python3 -m schedule_risk_agent.training_pipeline rollback \
        --output-root model_artifacts/schedule-risk \
        --model-version <model-version>

The legacy schedule_risk_agent.train command remains a proof-of-concept
compatibility command and must not create a production release.

## Runtime

Set MODEL_BUNDLE_PATH to load one verified release bundle. The older MODEL_PATH,
FEATURE_SCHEMA_PATH, and MODEL_CARD_PATH settings remain a development fallback.

Run the MCP service in the Python 3.11 container when Docker access is available:

    docker compose -f docker-compose.schedule-risk.yml up schedule-risk-agent

The local publisher and repository are interim storage implementations. The
production design publishes to and reads from the persistent Snowflake
SCHEDULE_PROJECT_FEATURES_CURRENT table.

