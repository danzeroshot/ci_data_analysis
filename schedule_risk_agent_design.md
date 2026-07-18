# Schedule Risk Agent Design

## Purpose

The Schedule Risk Agent scores project schedule-delay risk at the project level. It is designed as a Docker-containerized HTTP/SSE MCP server backed by a serialized Python/scikit-learn model. The first production-oriented model should use beginning-of-project features only: planned schedule, project scope/complexity, planned value, item price/quantity distributions, approved keyword features, and other fields expected to be present after project/contract/item setup but before payments.

The agent predicts one of three schedule-risk bins:

| Bin | Label | Definition |
| --- | --- | --- |
| 0 | `no_delay` | `PercentDelayed <= 0` |
| 1 | `mild_delay` | `0 < PercentDelayed <= 25` |
| 2 | `significant_delay` | `PercentDelayed > 25` |

The retrospective training target is:

```text
PercentDelayed = 100.0 * TargetActualDurationDays / TargetPlannedDurationDays - 100.0
```

where the prior analysis used first valid posting date minus 30 days as the actual-start proxy and last valid posting date as the actual-end proxy. This target definition should be reviewed with Aurigo before production use.

## Design Principles

- The scoring grain is one project: `CustomerName + ProjectID`.
- The MCP caller provides project identifiers and a short-lived Snowflake authentication token. The agent extracts all required features itself.
- The model is customer-agnostic by default. `CustomerName` is used for database routing/filtering, not as a model feature.
- Retrospective payment/target fields are never used as inference features.
- Change-order features are excluded from the initial beginning-of-project agent.
- Every response includes model version, feature schema version, and feature extraction metadata.
- Batch scoring is supported to reduce per-project database overhead.

## Container Architecture

The schedule agent is deployed as its own Docker service/container.

Recommended service name:

```yaml
schedule-risk-agent
```

The same image may contain both serving and training code, but the runtime command determines behavior:

```text
python -m schedule_risk_agent.server   # MCP HTTP/SSE inference server
python -m schedule_risk_agent.train    # offline training pipeline
python -m schedule_risk_agent.evaluate # scoring/performance audit
```

### Container Contents

```text
/app
  schedule_risk_agent/
    server.py                 # HTTP/SSE MCP server
    mcp_tools.py              # MCP tool definitions and payload validation
    feature_extract.py        # Snowflake feature extraction queries
    feature_schema.py         # expected feature list, types, defaults
    model_loader.py           # model artifact loading and version checks
    train.py                  # offline training entrypoint
    evaluate.py               # scoring/performance entrypoint
    errors.py                 # normalized error responses
  models/
    schedule_risk_model.pkl
    schedule_risk_feature_schema.json
    schedule_risk_model_card.json
  sql/
    schedule_feature_extract.sql
    schedule_training_extract.sql
  pyproject.toml or requirements.txt
```

### Docker Compose Sketch

```yaml
services:
  schedule-risk-agent:
    build:
      context: .
      dockerfile: docker/schedule-risk-agent.Dockerfile
    image: aurigo/schedule-risk-agent:latest
    command: ["python", "-m", "schedule_risk_agent.server"]
    ports:
      - "8011:8011"
    environment:
      MCP_HOST: "0.0.0.0"
      MCP_PORT: "8011"
      MODEL_PATH: "/app/models/schedule_risk_model.pkl"
      FEATURE_SCHEMA_PATH: "/app/models/schedule_risk_feature_schema.json"
      MODEL_CARD_PATH: "/app/models/schedule_risk_model_card.json"
      SNOWFLAKE_ACCOUNT: "${SNOWFLAKE_ACCOUNT}"
      SNOWFLAKE_WAREHOUSE: "${SNOWFLAKE_WAREHOUSE}"
      SNOWFLAKE_ROLE: "${SNOWFLAKE_ROLE}"
      SNOWFLAKE_DATABASE: "${SNOWFLAKE_DATABASE}"
      SNOWFLAKE_SCHEMA: "${SNOWFLAKE_SCHEMA}"
    healthcheck:
      test: ["CMD", "python", "-m", "schedule_risk_agent.healthcheck"]
      interval: 30s
      timeout: 5s
      retries: 3
```

Port `8011` is exposed outside Docker so MCP clients outside the container can connect to the HTTP/SSE server.

## MCP Interface

Transport: HTTP/SSE MCP server.

The server exposes one primary tool:

```text
score_schedule_risk
```

Optional administrative tools:

```text
get_schedule_model_metadata
validate_schedule_feature_extraction
```

### Tool: `score_schedule_risk`

Scores one or more projects for schedule-delay risk.

#### Input Payload

```json
{
  "request_id": "req-2026-06-19-0001",
  "auth": {
    "type": "snowflake_oauth_token",
    "access_token": "<short-lived-token>",
    "expires_at_utc": "2026-06-19T15:30:00Z"
  },
  "snowflake": {
    "account": "optional override; otherwise env var",
    "warehouse": "optional override; otherwise env var",
    "role": "optional override; otherwise env var",
    "database": "optional override; otherwise env var",
    "schema": "optional override; otherwise env var"
  },
  "projects": [
    {
      "customer_name": "UDOT",
      "project_id": "1425"
    },
    {
      "customer_name": "Lincoln",
      "project_id": "12345"
    }
  ],
  "options": {
    "include_feature_values": false,
    "include_top_drivers": true,
    "include_debug_metadata": false
  }
}
```

#### Input Rules

- `auth.access_token` is required and must be short-lived.
- `projects` must contain at least one project.
- Batch size should be capped by configuration, for example `MAX_BATCH_SIZE=500`.
- `customer_name` and `project_id` are required for every project.
- The caller does not provide feature values in the first production design.

#### Return Payload

```json
{
  "request_id": "req-2026-06-19-0001",
  "agent": "schedule-risk-agent",
  "model_version": "schedule-rf-2026-06-19.1",
  "feature_schema_version": "project-approved-keywords-2026-06-11",
  "training_data_snapshot_id": "snowflake-project-features-2026-06-11",
  "scored_at_utc": "2026-06-19T15:00:00Z",
  "results": [
    {
      "customer_name": "UDOT",
      "project_id": "1425",
      "status": "scored",
      "prediction": {
        "risk_bin": 1,
        "risk_label": "mild_delay",
        "risk_score": 0.64,
        "class_probabilities": {
          "no_delay": 0.22,
          "mild_delay": 0.64,
          "significant_delay": 0.14
        }
      },
      "top_drivers": [
        {
          "feature": "DOLLARSPERPLANNEDMONTH",
          "value": 124000.0,
          "direction": "higher_values_associated_with_higher_delay_risk",
          "importance_rank": 1
        }
      ],
      "feature_extraction": {
        "feature_row_found": true,
        "missing_required_feature_count": 0,
        "imputed_feature_count": 14,
        "feature_extracted_at_utc": "2026-06-19T15:00:00Z"
      }
    }
  ],
  "errors": []
}
```

### Error Responses

Top-level transport or request errors should return a standard MCP error. Per-project errors should be returned inside `results` when batch scoring partially succeeds.

#### Request-Level Error

```json
{
  "request_id": "req-2026-06-19-0001",
  "agent": "schedule-risk-agent",
  "status": "failed",
  "error": {
    "code": "AUTH_TOKEN_EXPIRED",
    "message": "The supplied Snowflake access token is expired or too close to expiration.",
    "retryable": true,
    "details": {
      "expires_at_utc": "2026-06-19T14:59:00Z"
    }
  }
}
```

#### Per-Project Error

```json
{
  "customer_name": "UDOT",
  "project_id": "1425",
  "status": "not_scored",
  "error": {
    "code": "FEATURE_ROW_NOT_FOUND",
    "message": "No beginning-of-project feature row was found for CustomerName + ProjectID.",
    "retryable": false
  }
}
```

Recommended error codes:

| Code | Meaning | Retryable |
| --- | --- | --- |
| `INVALID_PAYLOAD` | Missing or invalid request fields | No |
| `BATCH_TOO_LARGE` | Project list exceeds configured limit | No |
| `AUTH_TOKEN_MISSING` | No short-lived token supplied | No |
| `AUTH_TOKEN_EXPIRED` | Token expired or close to expiration | Yes after token refresh |
| `SNOWFLAKE_CONNECTION_FAILED` | Could not connect to Snowflake | Yes |
| `SNOWFLAKE_QUERY_FAILED` | Feature query failed | Usually yes |
| `FEATURE_ROW_NOT_FOUND` | Project not found in feature extraction query | No |
| `FEATURE_SCHEMA_MISMATCH` | Extracted features do not match model schema | No until deployment fixed |
| `MODEL_LOAD_FAILED` | Serialized model cannot be loaded | No until deployment fixed |
| `MODEL_INFERENCE_FAILED` | Model prediction raised an exception | Depends |
| `INTERNAL_ERROR` | Unexpected server error | Depends |

## Feature Extraction Design

The inference query should reproduce the approved beginning-of-project feature table for the requested project keys. The SQL should be derived from the prior project feature extraction lineage, but stripped of retrospective target fields and payment-history dependencies.

Input project keys should be staged into a temporary Snowflake table or passed as a `VALUES` CTE:

```sql
WITH requested_projects AS (
    SELECT column1::STRING AS CustomerName, column2::STRING AS ProjectID
    FROM VALUES
        ('UDOT', '1425'),
        ('Lincoln', '12345')
),
...
SELECT *
FROM project_beginning_feature_extract f
JOIN requested_projects r
  ON r.CustomerName = f.CustomerName
 AND r.ProjectID = f.ProjectID;
```

Feature groups:

- Planned schedule fields from contract start/end dates.
- Planned value and item value distribution fields.
- Scope complexity fields such as contract count, item count, item containers, standard item prefixes.
- Budget linkage fields only if Aurigo confirms beginning availability.
- Approved keyword features from project, contract, and contract item text.

Excluded groups:

- `PERCENTDELAYED`.
- All `TARGET*` fields.
- Payment/work-posting fields.
- `CUSTOMERNAME` as a model feature.
- raw identifiers/descriptors as model features, except for feature extraction and audit.
- Change-order features for the first version.

## Training Pipeline

The offline training pipeline is packaged in the same Docker image but run separately from the MCP server.

### Step 1: Get Training Data

Inputs:

- Snowflake connection configuration.
- Short-lived or service-token authentication approved for offline training.
- Training extraction SQL.
- Approved keyword list and feature schema version.

Process:

1. Query the project feature table or generate it from source Snowflake tables.
2. Compute `PercentDelayed` retrospectively.
3. Create the 3-bin schedule target.
4. Filter to rows with valid target and sufficient beginning feature coverage.
5. Exclude leakage fields and disallowed identifiers.
6. Freeze output as a training snapshot with row counts, timestamp, source SQL hash, and feature schema hash.

### Step 2: Tune Model

Initial implementation: scikit-learn random forest classifier.

Recommended tuning search dimensions:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `min_samples_split`
- `max_features`
- `class_weight`

Validation strategies:

- Hash-based train/test split for stable benchmark.
- Time-based split for future-project generalization.
- Leave-one-customer-out as a transferability stress test, but not as a deployment-blocking metric for tiny customers.

Primary optimization metric:

- Three-bin balanced accuracy or macro F1.

Secondary metrics:

- One-vs-rest multiclass AUC.
- Confusion matrix.
- Per-class recall, especially for `significant_delay`.
- Calibration curves if probabilities are exposed as risk scores.

### Step 3: Run Scoring/Performance

The scoring audit should produce:

- Overall metrics by validation split.
- Per-class precision/recall/F1.
- Confusion matrices.
- Feature importance summary.
- Feature missingness/imputation summary.
- Model calibration diagnostics.
- Comparison against simple baselines, such as majority class and duration/value heuristic.

### Step 4: Serialize Model

Artifacts:

```text
schedule_risk_model.pkl
schedule_risk_feature_schema.json
schedule_risk_model_card.json
schedule_risk_training_metrics.json
schedule_risk_requirements.lock
```

The serialized artifact should include or be packaged with:

- fitted imputer/preprocessor;
- fitted classifier;
- ordered feature list;
- class label mapping;
- model version;
- training data snapshot ID;
- source SQL hash;
- Python and package versions.

## Inference Pipeline

1. Receive MCP request.
2. Validate request shape and batch size.
3. Extract and validate short-lived Snowflake token.
4. Open Snowflake connection using token and configured account/warehouse/role/database/schema.
5. Query beginning-of-project features for `CustomerName + ProjectID` batch.
6. Align returned features to serialized `feature_schema.json`.
7. Apply imputation/preprocessing exactly as during training.
8. Run model prediction.
9. Generate class probabilities and risk label.
10. Optionally compute top drivers from global model importances and project feature values.
11. Return MCP payload with per-project results, model metadata, extraction metadata, and errors.

## Observability And Operations

Log structured events:

- request received;
- auth validation result, without logging token contents;
- Snowflake query duration;
- number of projects requested/scored/not scored;
- feature missingness counts;
- model version and feature schema version;
- inference duration;
- error codes.

Metrics:

- request count;
- request latency;
- Snowflake query latency;
- scoring latency;
- feature extraction failure rate;
- per-project not-scored rate;
- model load success/failure;
- class distribution of predictions.

Security:

- Never log access tokens.
- Enforce token expiration checks.
- Use least-privilege Snowflake role.
- Restrict outbound network access to Snowflake endpoints where possible.
- Validate all identifiers before constructing SQL; prefer bound parameters or staged key tables.

## Open Design Review Items

- Confirm Snowflake is the production inference source.
- Confirm all before-only features are truly available at the beginning of the project.
- Confirm planned date fields are not overwritten by actual execution dates.
- Confirm the `PercentDelayed` target definition.
- Confirm final bin thresholds: `0%` and `25%`.
- Decide whether probabilities need calibration before being shown directly as risk scores.
