# Schedule Risk Agent Detailed Architecture

## 1. Purpose And Scope

This document turns `schedule_risk_agent_design.md` into an implementable architecture for the Schedule Risk Agent. The agent is a Python/scikit-learn service exposed through an HTTP/SSE MCP server and deployed in its own Docker container.

The main architectural change from the earlier design is that inference does not calculate thousands of project features on demand. Beginning-of-project features are rebuilt periodically in Snowflake and published to a current feature table. The MCP server performs a keyed batch read, validates freshness and schema compatibility, and passes the ordered feature matrix to the serialized model.

The external mechanism that invokes feature refresh every six hours is outside scope. This architecture defines the SQL entrypoints, contracts, validation, and publication behavior that the external scheduler must call.

### In Scope

- Schedule-risk MCP inference service.
- Three-bin scikit-learn model runtime.
- Precalculated Snowflake feature store.
- Full feature rebuild and atomic current-table publication.
- Optional rolling operational history.
- Batch feature extraction.
- Offline training and model serialization commands packaged in the container.
- Model, feature-schema, keyword-manifest, and training-snapshot versioning.
- Docker service topology, security, observability, validation, and failure behavior.
- Snowflake SQL files needed to calculate, publish, retain, and extract features.

### Out Of Scope

- The scheduler/orchestrator that calls the six-hour refresh.
- Change-order features.
- Synchronous feature recalculation during an MCP request.
- Customer-specific models.
- Final approval of Snowflake as the production data source.
- Canonical construction-specification phrase integration.
- UI presentation of the predictions.

## 2. Confirmed Architecture Decisions

| Decision | Selected Design |
| --- | --- |
| Deployment | Dedicated Schedule Risk Agent container |
| MCP transport | HTTP/SSE |
| Model runtime | Python and scikit-learn |
| Database | Snowflake, pending design-review confirmation |
| Authentication | Caller-provided short-lived Snowflake token |
| Request grain | Batch of `CustomerName + ProjectID` keys |
| Feature input | Agent queries all model features; caller provides no feature values |
| Feature calculation | Full rebuild every six hours |
| Publication | Staging/next table followed by atomic Snowflake `SWAP` |
| Current table | One row per `CustomerName + ProjectID` |
| History | Separate optional table and SQL path |
| History retention | Configurable, initially three months |
| Maximum feature age | 24 hours |
| Stale behavior | Reject scoring; do not synchronously recalculate |
| Database access | Direct SQL access through a least-privilege role |
| Unapproved “Maybe” fields | Calculate and retain, but exclude from model schema until approved |
| Change orders | Excluded from version 1 |
| Training code | Packaged in same image, invoked as a separate command |

## 3. System Context

```text
                         outside this architecture
                    +-------------------------------+
                    | six-hour scheduler/orchestrator|
                    +---------------+---------------+
                                    |
                                    | invokes SQL sequence
                                    v
+-------------------+     +-------------------------------+
| Aurigo customer   |     | Snowflake analytics database  |
| source schemas    +---->| setup feature calculation     |
+-------------------+     | NEXT -> atomic SWAP -> CURRENT|
                          | optional HISTORY               |
                          +---------------+---------------+
                                          ^
                                          | keyed batch SELECT
                                          | short-lived token
+-------------------+    HTTP/SSE MCP      |
| MCP client / host +----------------------+
+-------------------+                      |
                                          v
                          +-------------------------------+
                          | schedule-risk-agent container |
                          | payload validation            |
                          | Snowflake connector           |
                          | schema alignment/imputation   |
                          | scikit-learn classifier       |
                          +-------------------------------+
```

## 4. Component Architecture

### 4.1 MCP HTTP/SSE Server

Responsibilities:

- Expose `score_schedule_risk` and model-metadata tools.
- Validate request ID, authentication payload, project keys, options, and batch limits.
- Never log or persist the short-lived token.
- Split a permitted batch into database-fetch chunks when needed.
- Preserve input ordering in the result payload.
- Return partial per-project failures without failing successful projects.
- Report the model and feature versions used for every response.

Recommended limits:

```text
MAX_BATCH_SIZE=500
DB_FETCH_CHUNK_SIZE=100
MAX_FEATURE_AGE_HOURS=24
SNOWFLAKE_QUERY_TIMEOUT_SECONDS=60
```

The public batch limit and internal database chunk size are separate. A request may contain up to 500 projects, while the service can issue five 100-project reads to control memory and wide-row transfer size.

### 4.2 Snowflake Feature Repository

Responsibilities:

- Create a Snowflake connection with the caller’s OAuth token.
- Use account, warehouse, role, database, and schema from approved server configuration. Request-level overrides should be disabled in production unless there is a validated multi-environment requirement.
- Bulk insert requested keys into a temporary table using bound values.
- Execute a trusted extraction query whose feature column list comes from the local model feature schema, never from request input.
- Return Arrow/pandas-compatible typed data.
- Surface query ID and timing for diagnostics without returning sensitive SQL details to ordinary clients.

### 4.3 Feature Schema Adapter

The feature schema is a versioned JSON artifact packaged with the model. It defines:

```json
{
  "feature_schema_version": "schedule-project-features-v1",
  "keyword_manifest_version": "approved-keywords-2026-06-11",
  "ordered_features": ["..."],
  "feature_types": {"PLANNEDDURATIONDAYS": "float64"},
  "nullable_features": ["..."],
  "imputation_strategy": "serialized_pipeline",
  "excluded_unapproved_features": ["BUDGETPLANNEDVALUESUM", "..."]
}
```

Responsibilities:

- Verify `FeatureSchemaVersion` and `KeywordManifestVersion` from Snowflake.
- Select and order columns exactly as trained.
- Reject missing required columns with `FEATURE_SCHEMA_MISMATCH`.
- Ignore extra store columns not present in the model schema.
- Coerce expected numeric types and turn non-finite values into nulls for the serialized imputer.
- Count imputed fields for response metadata and monitoring.

### 4.4 Model Runtime

The serialized scikit-learn pipeline contains preprocessing/imputation and the three-class classifier. It is loaded once during container startup and treated as immutable until process restart or a controlled model-reload deployment.

Output terminology:

- `risk_bin`: predicted class number.
- `risk_label`: `no_delay`, `mild_delay`, or `significant_delay`.
- `class_probabilities`: probability vector returned by the classifier.
- `predicted_class_probability`: probability assigned to the selected class.

The generic term `risk_score` should not be used unless a later calibrated business score is explicitly defined. Raw random-forest probabilities are not automatically calibrated probabilities of real-world outcomes.

### 4.5 Offline Training Command

Training code is present in the same image but is not reachable through the public MCP tool.

```text
python -m schedule_risk_agent.train
python -m schedule_risk_agent.evaluate
```

The training command requires a separately authorized Snowflake role/token and writes model artifacts to a controlled output volume or model registry. The inference container should mount released artifacts read-only.

## 5. Snowflake Feature Store

### 5.1 Object Layout

Default location:

```text
<ANALYTICS_DATABASE>.ML_FEATURES
```

Objects:

| Object | Purpose |
| --- | --- |
| `SCHEDULE_PROJECT_FEATURES_CURRENT` | Latest atomically published feature row per project |
| `SCHEDULE_PROJECT_FEATURES_NEXT` | Temporary physical build used for validation and swap |
| `SCHEDULE_PROJECT_FEATURES_HISTORY` | Optional rolling operational snapshots |
| `SCHEDULE_FEATURE_STORE_CONFIG` | Refresh, freshness, retention, and schema defaults |
| `SCHEDULE_FEATURE_REFRESH_AUDIT` | Refresh run status and validation metrics |
| `schedule_project_features_build` | Session-temporary calculated feature result |

### 5.2 Current Table Contract

Grain:

```text
one row per CustomerName + ProjectID
```

The physical table is wide because the model consumes explicit numeric columns. Keeping columns explicit provides Snowflake typing, column pruning, schema comparison, and direct loading into a DataFrame. The application must not issue an unbounded `SELECT *` in production; it builds a trusted projection from `schedule_risk_feature_schema.json` plus required metadata.

Required metadata columns:

| Column | Purpose |
| --- | --- |
| `CustomerName` | Routing and project key; excluded from the model matrix |
| `ProjectID` | Project key; excluded from the model matrix |
| `FeatureAsOfUtc` | Time represented by the calculated values |
| `FeatureSchemaVersion` | Feature transformation contract |
| `KeywordManifestVersion` | Approved keyword vocabulary contract |
| `RefreshRunId` | Atomic refresh identity |
| `RefreshStartedAtUtc` | Refresh operational metadata |

The store may contain raw identifiers, descriptions, status, “Maybe” fields, and other audit columns. Only columns in the released model feature schema enter inference.

### 5.3 Beginning-Feature Boundary

The production calculation excludes:

- payment rows;
- `PERCENTDELAYED` and all `TARGET*` fields;
- change orders;
- customer name as a model predictor;
- current project status as a model predictor;
- synchronous or request-time features.

The production text transformation intentionally differs from the earlier POC SQL in two places:

1. Project keyword text uses project name and project description, not current project status.
2. Item keyword text uses contract-item description, not linked budget-item description.

These changes remove inputs whose beginning availability is not established. The production model must be retrained using this exact calculation; the prior serialized POC model must not be served against the revised feature definition.

Contract keywords continue to use contract name and description.

### 5.4 Approved Keyword Shape

The generated calculation uses:

| Family | Approved keyword groups | Feature columns per group |
| --- | ---: | ---: |
| Project | 601 | 1 count |
| Contract | 477 | count and contract share |
| Item | 862 | count, item share, and planned-value share |

The generated SQL derives these lists from `approved_keyword_feature_column_filter_detail_2026-06-11.csv`. Any keyword change requires a new manifest version, regenerated SQL, a new feature schema, and model retraining.

## 6. Feature Refresh Architecture

### 6.1 Trigger Contract

An external scheduler invokes the following in one Snowflake session every six hours:

1. `schedule_risk_feature_calculation.sql`
2. `schedule_risk_feature_store_refresh_current.sql`
3. Optionally, `schedule_risk_feature_store_history.sql`

The scheduler must not run overlapping refreshes. Use an orchestration lock or a Snowflake task configuration that prevents concurrent runs.

### 6.2 Full Rebuild Sequence

```text
calculate session-temporary setup intermediates
            |
            v
schedule_project_features_build
            |
            v
SCHEDULE_PROJECT_FEATURES_NEXT
            |
            +--> validate nonempty build
            +--> validate unique CustomerName + ProjectID
            +--> validate schema/version and expected row-count bounds
            |
            v
atomic SWAP with SCHEDULE_PROJECT_FEATURES_CURRENT
            |
            +--> write refresh audit
            +--> optional append to HISTORY
            +--> apply rolling retention
```

Readers see either the complete previous table or the complete new table. They never read a partially rebuilt table.

### 6.3 Validation Gates

Required before swap:

- Build row count is greater than zero.
- `CustomerName + ProjectID` is unique.
- Feature schema version is exactly the deployed calculation version.
- Keyword manifest version is exactly the expected version.
- Every supported customer is present unless an explicit maintenance override exists.
- Row count is within an agreed deviation from the previous current table, initially recommended as `-20%` to `+50%`.
- Required model-feature columns are present.
- No required feature is returned with an incompatible Snowflake type.

The supplied refresh SQL implements the nonempty and duplicate-key gates. Customer coverage, row-count drift, and model-schema comparison should be added once production thresholds and the released feature manifest are fixed.

### 6.4 Failure Behavior

If calculation or validation fails:

- Do not swap.
- Leave `CURRENT` unchanged and available.
- Record a failed refresh audit row through the scheduler’s failure handler.
- Alert operations.
- Continue scoring from the previous table until it exceeds 24 hours old.
- Once older than 24 hours, return `FEATURE_DATA_STALE` per affected project rather than scoring silently.

### 6.5 Bootstrap Behavior

The first successful refresh creates `CURRENT` by cloning `NEXT`, then performs the same swap path used by later runs. This keeps bootstrap and recurring publication behavior aligned.

## 7. Operational History And Training Snapshots

### 7.1 Rolling History

`SCHEDULE_PROJECT_FEATURES_HISTORY` is optional and deliberately managed by a separate SQL file. It can be removed from the invocation sequence without changing calculation, publication, or inference.

Default retention:

```text
HISTORY_RETENTION_MONTHS=3
```

Each published `RefreshRunId` is inserted once. Retention is driven by `SCHEDULE_FEATURE_STORE_CONFIG`.

### 7.2 Training Snapshot Retention

Three months is not sufficient to preserve beginning snapshots until many projects complete and receive a retrospective delay label. Therefore, rolling operational history must not be the authoritative training source.

For each training/model version, create an immutable training snapshot outside the rolling history lifecycle, for example:

```text
<ANALYTICS_DATABASE>.ML_TRAINING.SCHEDULE_FEATURE_SNAPSHOT_<VERSION>
```

or an equivalent versioned Parquet artifact in controlled object storage.

The model card records the immutable snapshot ID. That snapshot is retained according to model-governance policy, not the three-month operational retention setting.

## 8. Inference Flow

```text
1. MCP server validates request and batch size.
2. Auth adapter verifies token presence and declared expiration.
3. Repository opens Snowflake OAuth connection.
4. Requested keys are bulk-loaded into a temporary table.
5. Repository selects the trusted model feature projection from CURRENT.
6. Missing keys become per-project FEATURE_ROW_NOT_FOUND errors.
7. FeatureAsOfUtc is checked against the 24-hour maximum age.
8. Version metadata is compared with the loaded model artifacts.
9. Schema adapter orders/coerces the feature matrix.
10. Serialized pipeline imputes and predicts probabilities.
11. Service returns labels, class probabilities, predicted-class probability,
    freshness metadata, and per-project errors.
```

No database token is written to logs, response payloads, traces, or model artifacts.

## 9. Freshness And Availability

Expected cadence:

```text
6 hours
```

Maximum acceptable feature age:

```text
24 hours
```

The six-hour cadence allows up to three consecutive failed refresh opportunities before inference must reject stale data. The MCP response includes:

- `feature_as_of_utc`;
- `feature_age_hours`;
- `refresh_run_id`;
- `feature_schema_version`;
- `keyword_manifest_version`.

Recommended service behavior:

| Feature state | Behavior |
| --- | --- |
| Row exists and age `<=24h` | Score normally |
| Row exists and age `>24h` | Per-project `FEATURE_DATA_STALE`; do not score |
| Row missing | Per-project `FEATURE_ROW_NOT_FOUND` |
| Version mismatch | Request-level `FEATURE_SCHEMA_MISMATCH` and readiness failure |

## 10. Security Architecture

### 10.1 Short-Lived Token

- The caller sends a short-lived Snowflake OAuth token in the MCP tool payload.
- The server validates presence, format, and declared expiration before connection.
- The token is held only in request memory.
- The token is not cached between requests.
- Logs redact the complete `auth` object.
- Connection pooling must not allow one caller’s authenticated session to be reused by another caller. Either disable pooling for token-authenticated connections or key pools by a nonreversible token fingerprint and expiration with strict eviction.

### 10.2 Snowflake Role

The inference role requires only:

- warehouse usage;
- analytics database/schema usage;
- `SELECT` on `SCHEDULE_PROJECT_FEATURES_CURRENT`;
- `SELECT` on `SCHEDULE_FEATURE_STORE_CONFIG` if configuration is read at inference.

It does not need access to customer source schemas, history, training labels, refresh DDL, or model artifacts.

The refresh role separately needs source-schema reads and create/alter/drop/insert privileges in `ML_FEATURES`.

### 10.3 SQL Safety

- Project keys are inserted with bound parameters or connector bulk methods.
- Database/schema/table names come from server configuration, not request payloads.
- The feature column projection comes from the signed/released local schema artifact.
- Customer names are validated against an allowlist.

## 11. Container And Deployment Architecture

### 11.1 Runtime Container

```text
schedule-risk-agent
  /app/schedule_risk_agent
  /app/sql
  /app/models (read-only released artifacts)
  port 8011
```

Startup sequence:

1. Load model, schema, and model card.
2. Validate artifact hashes and compatible package versions.
3. Verify model class mapping.
4. Start HTTP/SSE MCP server.
5. Mark liveness healthy after event loop starts.
6. Mark readiness healthy only after artifacts pass validation. Snowflake is checked on demand because credentials are caller-provided.

### 11.2 Training Invocation

The same image is run with a different command and a writable artifact output mount:

```yaml
command: ["python", "-m", "schedule_risk_agent.train"]
```

Training is not a long-running service and is not exposed on port 8011.

### 11.3 Model Release

Recommended artifact set:

```text
schedule_risk_model.joblib
schedule_risk_feature_schema.json
schedule_risk_model_card.json
schedule_risk_training_metrics.json
requirements.lock
checksums.sha256
```

Model activation is a container/image release, not an in-place file overwrite. Rollback deploys the prior image/artifact version.

## 12. Training Architecture

### 12.1 Target

```text
PercentDelayed =
    100 * TargetActualDurationDays / TargetPlannedDurationDays - 100
```

Bins:

| Class | Rule |
| --- | --- |
| `no_delay` | `PercentDelayed <= 0` |
| `mild_delay` | `0 < PercentDelayed <= 25` |
| `significant_delay` | `PercentDelayed > 25` |

The target definition remains a design-review item because actual start is proxied by first posting minus 30 days.

### 12.2 Pipeline

1. Select an immutable beginning-feature snapshot.
2. Calculate retrospective labels in a separate target query.
3. Join labels to features by `CustomerName + ProjectID`.
4. Exclude identifiers, target fields, unapproved “Maybe” fields, and change-order fields.
5. Split by stable hash, time, and leave-one-customer-out stress tests.
6. Tune regularized random-forest parameters.
7. Evaluate balanced accuracy, macro F1, one-vs-rest AUC, per-class recall, confusion matrix, and calibration.
8. Fit the released pipeline on the approved training population.
9. Serialize artifacts and write a model card.
10. Run an inference-parity test against the current feature-extraction code.

### 12.3 Mandatory Parity Test

For a fixed project sample, compare:

- training snapshot feature names/order/types;
- current feature-table extraction;
- Python schema adapter output;
- serialized pipeline input.

Release fails if any required feature differs unexpectedly or if transformations are implemented separately in Python and SQL with different semantics.

## 13. MCP Response Contract Adjustments

The detailed MCP payload remains defined in `schedule_risk_agent_design.md`, with these architecture clarifications:

- Rename `risk_score` to `predicted_class_probability`.
- Always include the complete `class_probabilities` mapping.
- Include `feature_as_of_utc`, `feature_age_hours`, and `refresh_run_id` per project.
- Include top drivers only when explicitly requested.
- Do not describe global impurity importance as a project-specific causal explanation. A later explanation implementation should use a documented local method.

Additional error codes:

| Code | Scope | Meaning |
| --- | --- | --- |
| `FEATURE_DATA_STALE` | Project | Feature row is older than 24 hours |
| `FEATURE_VERSION_MISMATCH` | Request | Store and model feature versions differ |
| `KEYWORD_MANIFEST_MISMATCH` | Request | Store and model keyword manifests differ |
| `PARTIAL_BATCH_FAILURE` | Request summary | Some projects scored and some failed |

## 14. Observability

### Service Metrics

- MCP request count and latency.
- Projects requested, scored, missing, stale, and failed.
- Batch size and internal chunk count.
- Snowflake connection and query latency.
- Feature rows transferred and bytes transferred.
- Imputed feature count distribution.
- Prediction class distribution.
- Model/schema/manifest mismatch count.

### Refresh Metrics

- Refresh start/end/duration.
- Build and published row counts.
- Customer coverage.
- Duplicate project keys.
- Row-count change versus previous refresh.
- Feature null-rate changes for monitored columns.
- Current table age.
- History append and retention-delete counts.

### Logging

Structured logs include request ID, Snowflake query ID, refresh run ID, model version, schema version, counts, durations, and normalized error codes. They exclude tokens, raw descriptions, and full feature vectors.

## 15. SQL Deliverables

| File | Responsibility |
| --- | --- |
| `schedule_risk_feature_calculation.sql` | Calculate approved setup-only project features in a temporary build table |
| `schedule_risk_feature_store_ddl.sql` | Create schema control/config/audit objects and baseline grants |
| `schedule_risk_feature_store_refresh_current.sql` | Build NEXT, validate, and atomically publish CURRENT |
| `schedule_risk_feature_store_history.sql` | Independently append and retain optional three-month operational history |
| `schedule_risk_feature_store_extract.sql` | Batch inference extraction and freshness calculation |
| `schedule_risk_training_extract.sql` | Keep retrospective target join separate from inference features |
| `generate_schedule_risk_feature_store_sql.py` | Reproduce calculation SQL from the source query and approved manifest |

The SQL contains deployment placeholders such as `<ANALYTICS_DATABASE>` and role/table names. These must be resolved through environment-specific deployment configuration and reviewed in Snowflake before execution.

## 16. Testing Strategy

### SQL Tests

- Calculation returns one row per project key.
- Supported-customer coverage matches source expectations.
- Approved keyword column count matches the manifest.
- No `TARGET*` or `PERCENTDELAYED` column exists in current features.
- Project keyword tokenization excludes project status.
- Item keyword tokenization excludes budget-item description.
- Atomic swap leaves readers with a complete table during a rebuild.
- Failed validation leaves current unchanged.
- History insertion is idempotent by refresh run.
- Retention setting changes delete the expected snapshots.
- Batch extraction returns missing keys and freshness state correctly.

### Python Tests

- Payload and batch validation.
- Token expiration and redaction.
- Feature column ordering and type coercion.
- Schema/manifest mismatch rejection.
- Stale and missing per-project errors.
- Partial batch success.
- Model class mapping and probability sum.
- Deterministic model loading and artifact checksum validation.

### Integration Tests

- Container-to-Snowflake connection with a short-lived test token.
- End-to-end batch score against a controlled feature table.
- Token isolation across concurrent callers.
- 500-project request with 100-project database chunks.
- Current-table swap while scoring requests are active.
- Rollback to the prior model image.

## 17. Implementation Sequence

1. Review and approve production feature boundary with Aurigo.
2. Resolve Snowflake database, warehouse, role, and schema placeholders.
3. Run generated calculation SQL in a development Snowflake environment.
4. Compare production-safe features with the approved CSV and explain expected differences from text-boundary changes.
5. Deploy control objects and perform first atomic current-table bootstrap.
6. Add optional history and confirm three-month retention behavior.
7. Implement repository and schema-adapter code.
8. Capture an immutable training snapshot from the production-safe calculation.
9. Retrain and evaluate the three-bin model.
10. Package signed/versioned model artifacts in the container.
11. Implement MCP tool and error contracts.
12. Run concurrency, freshness, security, and swap integration tests.
13. Connect the external six-hour scheduler.
14. Conduct design review and production-readiness review.

## 18. Remaining Review Items

- Confirm Snowflake is the final online feature source.
- Confirm account/database/schema and role names.
- Confirm all released model features are available at project beginning.
- Decide whether the five budget-linkage fields and `NUMCOMMITMENTS` can enter a later model schema.
- Confirm the schedule target and 0%/25% bin thresholds.
- Set customer-coverage and row-count drift thresholds for refresh validation.
- Define immutable training-snapshot retention and model-governance policy.
- Decide whether class probabilities require calibration before client presentation.

