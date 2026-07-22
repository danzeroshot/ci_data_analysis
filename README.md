# Aurigo Schedule and Budget Risk Modeling

This repository contains the project-level risk analysis, Snowflake feature SQL, schedule-risk training pipeline, model runtime, and HTTP/SSE MCP server.

The operational implementation currently covers schedule risk. Budget-risk design documentation exists, but a production budget-risk training and runtime implementation is not yet present.

## Repository layout

    schedule_risk_agent/          Python runtime, feature refresh, MCP server
    schedule_risk_agent/training_pipeline/
                                  snapshots, qualification, tuning, releases
    config/                       training and release configuration
    models/                       development model/schema artifacts
    model_artifacts/              versioned runs, candidates, and releases
    feature_snapshots/            local immutable feature snapshots
    training_snapshots/           immutable label snapshots
    data_analysis/                exploratory SQL, CSVs, notebooks, and reports
    docker/                       container build definitions
    docker-compose.schedule-risk.yml
                                  schedule-risk services
    tests/                        unit, runtime, MCP, and training tests
    requirements.lock             locked Python dependencies

## Prerequisites

Use Python 3.11 when possible:

    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install --requirement requirements.lock

Run the test suite from the repository root:

    python3 -m pytest -q

Snowflake workflows require an account, user, role, warehouse, short-lived programmatic access token, and permission to create temporary tables. The default local configuration is read from snowflake_access_config.txt.txt and the token from schedule_model_dev-token-secret.txt. Keep both files local and never commit them.

## Operational flow

The intended flow is:

1. Execute the approved Snowflake feature SQL.
2. Validate the one-row-per-project feature build.
3. Publish an immutable local Parquet snapshot, or publish to the persistent Snowflake feature table after deployment privileges are available.
4. Create an immutable label snapshot.
5. Generate or review the candidate feature manifest.
6. Run deterministic regularized random-forest training.
7. Review metrics, subgroup gates, customer dispositions, and release gates.
8. Promote verified global and customer-specific bundles.
9. Run local inference or the MCP service.

The schedule target has three classes:

    0: no delay
    1: mild delay
    2: significant delay

The current label definition is:

    PERCENTDELAYED <= 0       -> class 0
    0 < PERCENTDELAYED <= 25  -> class 1
    PERCENTDELAYED > 25       -> class 2

## Feature generation

The approved beginning-available feature SQL is:

    schedule_risk_feature_calculation.sql

It creates the staged/materialized project feature intermediates and ends with the feature build expected by the refresh program. It is intended to run in one Snowflake session because temporary tables are session-scoped. It uses LPS.STATUSNAME for project status and excludes target/post-payment fields from predictors.

The model feature schema is:

    models/schedule_risk_feature_schema.json

Generate the candidate feature manifest:

    python3 -m schedule_risk_agent.training_pipeline generate-manifest \
        --schema models/schedule_risk_feature_schema.json \
        --output config/schedule_candidate_features.json

### Local feature refresh

The development refresh connects to Snowflake, executes the feature SQL, validates the temporary build table, and publishes a local snapshot:

    python3 -m schedule_risk_agent.feature_refresh --target local

Snapshots are written below:

    feature_snapshots/snapshots/<build-id>/
    feature_snapshots/current.json

Each snapshot contains Parquet data, a manifest, validation metadata, checksums, and a COMPLETE marker. The runtime rejects stale, incomplete, tampered, duplicate-key, or schema-incompatible snapshots.

Important feature environment variables include:

    SNOWFLAKE_CONFIG_FILE
    SNOWFLAKE_ACCOUNT
    SNOWFLAKE_USER
    SNOWFLAKE_ROLE
    SNOWFLAKE_WAREHOUSE
    SNOWFLAKE_DATABASE
    SNOWFLAKE_SCHEMA
    SNOWFLAKE_TOKEN_FILE
    FEATURE_SQL_PATH
    FEATURE_SCHEMA_PATH
    FEATURE_SNAPSHOT_ROOT

The persistent Snowflake publisher is intentionally not enabled until the required client privileges and deployment table are available. The local snapshot path is the current development substitute.

## Label snapshots

Create a development-only snapshot from the approved historical CSV:

    python3 -m schedule_risk_agent.training_pipeline snapshot-labels \
        --source-csv data_analysis/custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv \
        --output-root training_snapshots/labels \
        --target-version schedule-delay-v1

The snapshot records labels, exclusions, source hashes, class counts, and checksums. Missing keys, invalid targets, and duplicate project keys are excluded with reasons. CSV-derived labels are permanently marked development_only and cannot pass production lineage validation.

When the production Snowflake label SQL and permissions are ready:

    python3 -m schedule_risk_agent.training_pipeline snapshot-labels-snowflake \
        --sql schedule_risk_label_calculation.sql \
        --output-root training_snapshots/labels \
        --target-version schedule-delay-v1

Review the SQL and reconciliation before treating a Snowflake label snapshot as production eligible.

## Training pipeline

The main development configuration is:

    config/schedule_training_run.development.json

It enables regularized random-forest tuning, deterministic hash splits, cross-validation, temporal evaluation, required subgroup-family evaluation, customer-specific models, and HTML reporting.

Qualification only:

    python3 -m schedule_risk_agent.training_pipeline qualify \
        --config config/schedule_training_run.development.json

Qualification and tuning:

    python3 -m schedule_risk_agent.training_pipeline tune \
        --config config/schedule_training_run.development.json

Complete training run:

    python3 -m schedule_risk_agent.training_pipeline run \
        --config config/schedule_training_run.development.json

Runs are written to:

    model_artifacts/schedule-risk/runs/<run-id>/

Important artifacts include:

    run_result.json
    run_config.json
    qualified_feature_schema.json
    selected_hyperparameters.json
    locked_holdout_metrics.json
    cv_out_of_fold_predictions.parquet
    locked_holdout_predictions.parquet
    metrics/metrics_long.parquet
    subgroups/subgroup_eligibility.json
    subgroups/subgroup_metrics_long.parquet
    customer_models/customer_model_eligibility.json
    candidate/
    report.html
    plots/

### Subgroup evaluation

The global model is evaluated across:

- planned duration: less than 60, 60 through 364, and at least 365 days;
- planned value: at most 0, greater than 0 through less than 1 million, 1 million through less than 10 million, and at least 10 million;
- contract item count: 1 through 19, 20 through 49, and at least 50;
- qualified-predictor missingness: 0 percent and greater than 0 percent.

Every configured band must meet development/CV and locked-holdout row and class-support thresholds. If one required family fails, the candidate is rejected for new global production promotion and the incumbent remains active.

### Customer-specific models

A customer is independently tuned when its customer development and holdout support passes the tuning thresholds. When only the absolute floor passes, the customer model is fitted using the selected global hyperparameters. Below the floor, no customer model is fitted and the result is explicitly unavailable.

The all-customer model is mandatory. Inference returns both outputs and does not choose between them.

## Releases

Validate and promote a global host release:

    python3 -m schedule_risk_agent.training_pipeline release \
        --run-dir model_artifacts/schedule-risk/runs/<run-id> \
        --output-root model_artifacts/schedule-risk

The global pointer is:

    model_artifacts/schedule-risk/current.json

Production promotion adds the production flag:

    python3 -m schedule_risk_agent.training_pipeline release \
        --run-dir model_artifacts/schedule-risk/runs/<run-id> \
        --output-root model_artifacts/schedule-risk \
        --production

Production requires approved numeric thresholds, production-eligible labels, serialization parity, and passing required subgroup-family eligibility.

Promote one customer candidate:

    python3 -m schedule_risk_agent.training_pipeline customer-release \
        --candidate <run-dir>/customer_models/UDOT/candidate \
        --customer UDOT \
        --output-root model_artifacts/schedule-risk

Customer pointers are stored under:

    model_artifacts/schedule-risk/customer-releases/<customer>/current.json

Customer models must reference a released parent all-customer model. A customer failure does not block other customers or the global model.

Rollback the global pointer:

    python3 -m schedule_risk_agent.training_pipeline rollback \
        --output-root model_artifacts/schedule-risk \
        --model-version <model-version>

Compare a released bundle with later immutable data:

    python3 -m schedule_risk_agent.training_pipeline compare \
        --bundle <release-directory> \
        --features <features.parquet> \
        --feature-manifest <feature-manifest.json> \
        --labels <labels.parquet> \
        --label-manifest <label-manifest.json> \
        --output <evaluation-directory>

## Local model runtime

The preferred runtime loads a verified bundle:

    export MODEL_BUNDLE_PATH="$PWD/model_artifacts/schedule-risk/releases/<model-version>"
    export CUSTOMER_MODEL_ROOT="$PWD/model_artifacts/schedule-risk/customer-releases"
    export FEATURE_SNAPSHOT_ROOT="$PWD/feature_snapshots"
    export FEATURE_REPOSITORY=local
    export MCP_HOST=127.0.0.1
    export MCP_PORT=8011

    python3 -m schedule_risk_agent.server

The legacy development fallback uses MODEL_PATH, MODEL_CARD_PATH, and FEATURE_SCHEMA_PATH. Bundle loading is preferred because it validates checksums, schema, model classes, and serialized prediction parity.

Inference keys are CUSTOMERNAME plus PROJECTID. Feature snapshots are rejected when older than MAX_FEATURE_AGE_HOURS, which defaults to 24. MAX_BATCH_SIZE defaults to 500.

## MCP HTTP/SSE server

The server exposes:

    GET  /health/live
    GET  /health/ready
    GET  /sse
    POST /messages
    POST /mcp

The tool is:

    score_schedule_risk

Basic health checks:

    curl http://127.0.0.1:8011/health/live
    curl http://127.0.0.1:8011/health/ready

List tools:

    curl -s http://127.0.0.1:8011/mcp \
      -H 'Content-Type: application/json' \
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

Score a batch:

    curl -s http://127.0.0.1:8011/mcp \
      -H 'Content-Type: application/json' \
      -d '{
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
          "name": "score_schedule_risk",
          "arguments": {
            "request_id": "smoke-test-1",
            "projects": [
              {"customer_name": "UDOT", "project_id": "12345"}
            ]
          }
        }
      }'

The response includes the all-customer result and, when available, the customer-specific result. An unavailable customer model is reported without failing the all-customer result or unrelated batch rows.

## Docker

Docker Compose exposes three independent, opt-in workflows using the same image:

- `schedule-risk-training`: one-shot training with read-only input snapshots and
  a writable host artifact directory;
- `schedule-risk-feature-refresh`: one-shot Snowflake refresh into the host
  feature snapshot directory; and
- `schedule-risk-agent`: long-running MCP inference with read-only feature and
  model mounts and no Snowflake credential.

Set the host identity before running any Compose workflow. The writers keep
generated files owned by the invoking user, and MCP can read their private
snapshot directories:

    export SCHEDULE_RISK_HOST_UID="$(id -u)"
    export SCHEDULE_RISK_HOST_GID="$(id -g)"

Build the shared image (the services are opt-in profiles, so select one service
explicitly):

    docker compose -f docker-compose.schedule-risk.yml build schedule-risk-agent

Run feature qualification only:

    docker compose -f docker-compose.schedule-risk.yml run --rm \
      schedule-risk-training python -m schedule_risk_agent.training_pipeline \
      qualify --config config/schedule_training_run.development.json

Run the complete training pipeline:

    docker compose -f docker-compose.schedule-risk.yml run --rm \
      schedule-risk-training

Training reads `config/`, `feature_snapshots/`, and `training_snapshots/`
read-only and writes new runs below `model_artifacts/`.

Refresh the local feature snapshot from Snowflake before starting MCP when the
current snapshot is missing or older than `MAX_FEATURE_AGE_HOURS`:

    docker compose -f docker-compose.schedule-risk.yml run --rm \
      schedule-risk-feature-refresh

The refresh process receives the local Snowflake config and PAT as Compose
secrets. The MCP process never mounts either secret.

Start only the MCP service:

    docker compose -f docker-compose.schedule-risk.yml up -d schedule-risk-agent

View logs:

    docker compose -f docker-compose.schedule-risk.yml logs -f schedule-risk-agent

The compose file binds the MCP port to loopback host port 8011 by default:

    http://127.0.0.1:8011

The MCP service uses baseline files under `/app/models` unless
`SCHEDULE_RISK_MODEL_BUNDLE_PATH` is set to a container path below the mounted
`/app/model_artifacts`, for example:

    export SCHEDULE_RISK_MODEL_BUNDLE_PATH=/app/model_artifacts/schedule-risk/releases/<model-version>

Customer release pointers are read from the mounted
`/app/model_artifacts/schedule-risk/customer-releases` directory.

Docker is not production-ready until released model/customer bundles, secret-manager integration, persistent feature-table refresh permissions, health/resource/network policies, and rollback procedures are validated.

## Exploratory analysis

The data_analysis directory contains the historical investigation:

- contract-item payment and burn EDA;
- burn-delta distribution and interval analysis;
- cumulative-spend curve characterization;
- Beta-CDF and linear model comparisons;
- proxy-label generation and diagnostics;
- project-level delay correlation analysis;
- random-forest proof-of-concept notebooks;
- change-order lifecycle analysis; and
- executive HTML/PDF report generation.

These notebooks are analytical evidence, not the production inference path. Open them with:

    jupyter notebook data_analysis/

Execute and export a notebook:

    jupyter nbconvert --to notebook --execute \
        --output executed.ipynb \
        data_analysis/<notebook>.ipynb

    jupyter nbconvert --to html data_analysis/executed.ipynb

The SQL under data_analysis is Snowflake-specific and may depend on customer schemas, temporary tables, session scope, and client permissions. Review each SQL file before running it against a different environment.

## Security and configuration

Never commit:

- Snowflake tokens;
- credential-bearing configuration;
- raw customer extracts;
- production release secrets; or
- unreviewed generated snapshots.

Use environment variables or deployment secret mounts. The application uses a short-lived Snowflake programmatic access token and does not place tokens in model bundles.

Feature schemas and model bundles are versioned contracts. Changing them requires a new training run and release candidate.

## Troubleshooting

No current feature snapshot:

    python3 -m schedule_risk_agent.feature_refresh --target local

Then check feature_snapshots/current.json and the snapshot COMPLETE marker.

Feature schema mismatch:
verify the feature snapshot manifest, model bundle schema, keyword manifest, and candidate schema. Do not bypass schema validation.

Rejected model:
inspect run_result.json, candidate/schedule_risk_model_card.json, and candidate/schedule_risk_training_metrics.json. Typical causes are failed release gates, development-only labels, missing artifacts, checksum failure, or parity failure.

MCP not ready:
check both health endpoints, feature snapshot age, current pointer, checksums, model bundle path, and service logs.

Snowflake refresh failure:
check token permissions, account configuration, role/warehouse access, database/schema defaults, temporary-table privileges, and whether the feature SQL ran in one session.

## Current status

The schedule-risk pipeline has been validated on the current 3,469-project development snapshot. All four required subgroup families passed eligibility. Customer dispositions were generated for Adams, Amtrak, CCD, CLV, Lincoln, and UDOT.

Production remains blocked by development-only label lineage and unapproved numeric performance thresholds. Customer release-pointer integration, final MCP schema documentation, persistent Snowflake feature-table deployment, and Docker validation remain deployment work.
