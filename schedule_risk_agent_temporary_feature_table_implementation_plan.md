# Schedule Risk Agent Implementation Plan
## Execution Status


Last updated: 2026-07-18

| Workstream | Status | Notes |
| --- | --- | --- |
| Inventory and dependency assessment | Complete | Existing SQL/notebooks reviewed; no released model artifacts found |
| Plan status tracking | Complete | Updated during implementation |
| Feature SQL correction and validation | Complete | Generator fixed, SQL regenerated, Python compile check passed |
| Shared refresh and local publication | Complete | Refresh, validation, atomic Parquet publisher implemented |
| Feature repositories and model runtime | Interim complete | Local repository and development model validated; persistent repository awaits table privileges |
| HTTP/SSE MCP service | Complete | Live initialize, tools/list, tools/call, health, and SSE handshake passed |
| Docker and offline commands | Implemented | Dockerfile, Compose, training and refresh commands added; Docker binary unavailable for image test |
| Automated tests | Complete | 6 unit/contract tests passing plus live HTTP smoke test |
| Snowflake local-snapshot refresh | Complete | 5,762 rows, 4,205 calculated columns, 43 MB Parquet snapshot published |
| Persistent Snowflake publication | Blocked | Requires client CREATE/ALTER privileges; production SQL retained |

## 1. Purpose

This plan keeps the intended production architecture intact while persistent
Snowflake table privileges are unavailable. Production uses a precalculated
Snowflake feature table. The interim implementation runs the same calculation
and validation, then publishes an immutable local Parquet snapshot.

The permission limitation changes only publication and retrieval:

1. A feature refresh job calculates, validates, and publishes features.
2. An HTTP/SSE MCP service reads published features and runs the model.

The MCP service does not calculate features, preserve a Snowflake session, or
depend on a temporary table. Both processes use one image with different
commands.

## 2. Core Decisions

- Refresh and inference are independent processes.
- Calculation and validation are identical for local and Snowflake targets.
- Local publication uses immutable Parquet snapshots and an atomic pointer.
- Production publication validates `NEXT` and atomically swaps it with `CURRENT`.
- MCP inference uses a configurable `FeatureRepository`.
- Local-to-Snowflake cutover is configuration, not a redesign.
- Training, refresh, and inference share one schema and SQL implementation.
- `CustomerName` is a lookup key, never a predictor.
- Targets, payments, and other leakage never enter inference features.
- Expected refresh is every six hours; maximum accepted age is 24 hours.

## 3. Model Contract

Grain: `CustomerName + ProjectID`.

| Bin | Label | Definition |
| ---: | --- | --- |
| 0 | `no_delay` | `PercentDelayed <= 0` |
| 1 | `mild_delay` | `0 < PercentDelayed <= 25` |
| 2 | `significant_delay` | `PercentDelayed > 25` |

```text
PercentDelayed =
    100.0 * TargetActualDurationDays / TargetPlannedDurationDays - 100.0
```

The current actual-start proxy is first valid posting date minus 30 days; actual
end is last valid posting date. The client must approve this target and the bins.

## 4. Architecture

```text
external scheduler
        |
        v
feature refresh command
  -> Snowflake temporary calculation
  -> common validation
  -> FeaturePublisher
       |-- LocalFeaturePublisher -> immutable Parquet snapshot
       +-- SnowflakeFeaturePublisher -> NEXT/SWAP/CURRENT

MCP client -> schedule-risk MCP -> FeatureRepository
                                  |-- LocalFeatureRepository
                                  +-- SnowflakeFeatureRepository
                                -> schema adapter
                                -> sklearn model
```

Production path:

```text
refresh -> SnowflakeFeaturePublisher -> SCHEDULE_PROJECT_FEATURES_CURRENT
MCP     -> SnowflakeFeatureRepository -> SCHEDULE_PROJECT_FEATURES_CURRENT
```

Interim path:

```text
refresh -> LocalFeaturePublisher -> versioned Parquet snapshot
MCP     -> LocalFeatureRepository -> versioned Parquet snapshot
```

## 5. Package Layout

```text
schedule_risk_agent/
  config.py
  server.py
  mcp_tools.py
  payloads.py
  errors.py
  telemetry.py
  feature_refresh.py
  feature_calculator.py
  feature_validation.py
  feature_schema.py
  publishers/{base,local,snowflake}.py
  repositories/{base,local,snowflake}.py
  model_loader.py
  predictor.py
  train.py
  evaluate.py
models/
sql/
tests/
docker/schedule-risk-agent.Dockerfile
docker-compose.yml
requirements.lock
```

## 6. Feature Refresh Pipeline

```text
python -m schedule_risk_agent.feature_refresh --target local
python -m schedule_risk_agent.feature_refresh --target snowflake
```

`FEATURE_PUBLISH_TARGET` provides the equivalent environment setting.

Every refresh:

1. Validate configuration and released feature metadata.
2. Read the Snowflake token from a secret file.
3. Open a dedicated refresh session.
4. Create a build ID and record start time.
5. Run `schedule_risk_feature_calculation.sql`.
6. Produce temporary `SCHEDULE_PROJECT_FEATURES_BUILD`.
7. Run common validation.
8. Invoke the configured publisher.
9. Verify publication and emit machine-readable results.
10. Close the Snowflake session.

The refresh connection is never owned by the MCP process.

### SQL Requirements

The final build has one row per project and includes `CustomerName`, `ProjectID`,
`FeatureAsOfUtc`, `FeatureSchemaVersion`, `KeywordManifestVersion`, and approved
model features. Source names remain fully qualified.

Execute SQL with a SQL-aware parser or generated statement manifest, not
`str.split(';')`.

Before implementation, fix the missing comma between the final expressions near
`ITEM_KW_ESCALATION_ITEM_SHARE` and
`ITEM_KW_ESCALATION_PLANNED_VALUE_SHARE`, including the generator.

### Validation Gates

- Nonzero row count.
- Unique project keys.
- Expected customer coverage.
- Matching schema and keyword versions.
- Every required model column with compatible type.
- No target, payment, change-order, or leakage columns.
- Valid timestamps and metadata.
- Approved row-count, null-rate, and numeric-range drift.
- Deterministic sample passes model preprocessing.

Validation writes JSON containing build ID, dimensions, coverage, versions,
checks, warnings, and pass/fail status. Warnings never bypass required gates.

## 7. Publisher Contract

```python
class FeaturePublisher(Protocol):
    def publish(self, build, validation) -> PublicationMetadata: ...
    def verify(self, publication) -> VerificationResult: ...
```

### Local Publisher

Snapshot layout:

```text
/var/lib/schedule-risk/features/
  snapshots/
    schedule-features-<build-id>/
      features.parquet
      manifest.json
      validation.json
      checksums.sha256
      COMPLETE
  current.json
```

Parquet is preferred to CSV because this table is very wide. It preserves types
and nulls, compresses sparse keyword columns, and supports projection and batch
reads. Use PyArrow with Zstandard; benchmark row-group and compression settings.

Atomic publication:

1. Create a hidden staging directory on the destination filesystem.
2. Select the validated build with a trusted column list.
3. Stream Snowflake result batches through `ParquetWriter`.
4. Require stable Arrow schema across batches.
5. Write metadata and checksums.
6. Reopen and verify row count, schema, and a sample.
7. Write `COMPLETE` last.
8. Atomically rename staging to its immutable build directory.
9. Fsync and atomically replace `current.json`.

A failed refresh cannot move the current pointer. Initially retain current plus
two prior successful snapshots.

The manifest records timestamps, dimensions, versions, SQL/schema hashes, model
compatibility, customer coverage, non-secret query lineage, validation status,
and file checksums.

### Snowflake Publisher

1. Create `SCHEDULE_PROJECT_FEATURES_NEXT` from the validated build.
2. Add refresh metadata and validate `NEXT`.
3. Bootstrap `CURRENT` if required.
4. Atomically swap `CURRENT` and `NEXT`.
5. Record audit metadata and optionally append history.
6. Drop the old `NEXT`.

Failed validation never changes `CURRENT`. Existing DDL, refresh, and history SQL
remain production deliverables but are unused in local mode.

## 8. Repository Contract

```python
class FeatureRepository(Protocol):
    def open(self) -> RepositoryMetadata: ...
    def fetch(self, keys, columns) -> FeatureBatch: ...
    def refresh_if_changed(self) -> bool: ...
    def health(self) -> RepositoryHealth: ...
    def close(self) -> None: ...
```

Select using `FEATURE_REPOSITORY=local|snowflake`. Repositories return keys,
metadata, missing rows, and trusted model columns; they do not preprocess or
score.

### Local Repository

At startup, read `current.json`, prevent path traversal, require `COMPLETE`,
verify checksums and versions, verify model columns, and run a smoke test.
Readiness remains false until this succeeds.

Support two measured access modes:

- `memory`: load trusted columns and index keys.
- `parquet`: PyArrow or DuckDB projection/filtering for bounded memory.

Do not parse the complete file for every request. Benchmark memory, startup, and
1/100/500-project lookup latency before selecting the default.

Poll `current.json`. Build and validate new repository state before atomically
swapping it in; in-flight readers finish on old state. Reject an invalid new
snapshot while retaining the prior valid one.

### Snowflake Repository

Use approved short-lived authentication, stage bound project keys, query
`SCHEDULE_PROJECT_FEATURES_CURRENT`, select trusted columns, return missing keys,
and enforce freshness/version compatibility. Persistent `CURRENT` remains
visible across request-scoped sessions.

## 9. MCP Contract

Primary tool: `score_schedule_risk`.

Input contains request ID, project keys, options, and `database_auth` only when
the configured Snowflake repository requires it. Local mode omits database
credentials. MCP transport authentication remains separate.

Each result includes project key/status, predicted bin/label, all three class
probabilities, predicted-class probability, model/schema/keyword versions,
feature build/time/age/repository, imputation counts, and normalized errors.

Do not call the output a generic `risk_score`. Raw random-forest probabilities
are not automatically calibrated real-world probabilities.

Core errors include `INVALID_PAYLOAD`, `BATCH_TOO_LARGE`,
`FEATURE_REPOSITORY_NOT_READY`, `FEATURE_SNAPSHOT_INVALID`,
`FEATURE_ROW_NOT_FOUND`, `FEATURE_DATA_STALE`,
`FEATURE_VERSION_MISMATCH`, `KEYWORD_MANIFEST_MISMATCH`,
`SNOWFLAKE_CONNECTION_FAILED`, `SNOWFLAKE_QUERY_FAILED`,
`FEATURE_PUBLICATION_FAILED`, `MODEL_LOAD_FAILED`,
`MODEL_INFERENCE_FAILED`, and `PARTIAL_BATCH_FAILURE`.

## 10. Model and Training

The released scikit-learn `Pipeline` contains deterministic column ordering,
imputation, transformations, and the tuned regularized random forest. Startup
verifies artifacts, versions, class order, repository compatibility, and a smoke
test before readiness.

Training is a separate command and:

1. Uses the same calculation or immutable validated snapshot.
2. Extracts labels separately and joins by project key.
3. Applies documented target-validity exclusions.
4. Excludes identifiers, customer, leakage, change orders, and unapproved fields.
5. Freezes data and lineage.
6. Runs hash, time, and customer-holdout evaluation.
7. Tunes regularized random-forest parameters.
8. Reports balanced accuracy, macro F1, recall, AUC, confusion, and calibration.
9. Compares majority and interpretable baselines.
10. Serializes model, schema, card, metrics, dependencies, and checksums.
11. Runs training/refresh/inference parity tests.

The production candidate is retrained on the production-safe calculation.

## 11. Docker and Configuration

Use one image with:

```text
python -m schedule_risk_agent.server
python -m schedule_risk_agent.feature_refresh --target local|snowflake
python -m schedule_risk_agent.train
python -m schedule_risk_agent.evaluate
```

In local mode, Compose defines a long-running MCP service and one-shot refresh
service sharing a named feature volume. MCP mounts it read-only. Only refresh
mounts the Snowflake secret.

Key refresh settings: `FEATURE_PUBLISH_TARGET`, Snowflake account/user/role/
warehouse, `SNOWFLAKE_TOKEN_FILE`, `FEATURE_SNAPSHOT_ROOT`, schema path, timeout,
and retention.

Key MCP settings: `FEATURE_REPOSITORY`, snapshot root/access mode, production
Snowflake table, batch/chunk limits, 24-hour age limit, pointer polling interval,
model path, and schema path.

Use Python 3.11 or 3.12, pinned dependencies, non-root containers, read-only
model artifacts, and production secret management.

## 12. Freshness, Observability, and Security

The scheduler runs refresh every six hours. A valid snapshot at most 24 hours old
is scored. An invalid new publication leaves the old one active. Once the active
publication exceeds 24 hours, scoring returns `FEATURE_DATA_STALE`. No valid
publication or version mismatch makes readiness false.

Refresh metrics cover query/build/export duration, query IDs, dimensions,
duplicates, coverage, validation, publication, age, and retention. MCP metrics
cover repository/build, reload, request latency, batch size, missing/stale rows,
lookup latency, imputation, and prediction distribution.

Logs exclude tokens, raw descriptions, vectors, and unbound SQL values.

Local refresh needs source `SELECT`, warehouse usage, and temporary-table
creation. Local MCP needs no Snowflake credential. Persistent roles add approved
publication or keyed `SELECT` privileges.

Controls include secret mounts, non-root execution, read-only MCP volume, path
validation, checksums, bound keys, trusted columns, customer allowlists, TLS, and
pinned/scanned dependencies.

## 13. Testing

- Unit-test configuration, redaction, SQL execution, validation, pointer
  atomicity, traversal, checksums, ordering, reload, payloads, and errors.
- SQL-test temporary-only calculation, grain, coverage, keyword counts, leakage,
  source qualification, and eventual persistent swap.
- Publisher-test null/type preservation, batch schema consistency, corruption,
  `COMPLETE`, failed publication, concurrent promotion, and retention.
- Run identical known/missing/mixed/max-batch, projection, freshness, version,
  and ordering fixtures against both repositories.
- Require equivalent `FeatureBatch` results for local and Snowflake backends.
- End-to-end test local refresh, MCP scoring, hot reload, corrupt publication,
  staleness, parity, and secret redaction.
- Benchmark calculation, export, size, memory, startup, lookup, and reload.

## 14. Implementation Phases

### Phase 0: SQL and Contract

Fix/regenerate the comma defect, run the complete calculation with current
privileges, verify coverage and features, approve target/bins, and measure output.

### Phase 1: Shared Refresh Core

Implement package/configuration, Snowflake calculation, common validation, and
build metadata.

### Phase 2: Local Publication

Implement streaming Parquet, manifests, checksums, immutable directories, atomic
pointer, verification, and retention.

### Phase 3: Local Repository and MCP

Implement repository contract, benchmark access strategy, model runtime, MCP
contracts, and hot reload.

### Phase 4: Model Release

Freeze production-safe data, tune/evaluate, version artifacts, and pass parity.

### Phase 5: Containerize and Harden

Package all commands and complete health, security, observability, integration,
and performance work.

### Phase 6: Persistent Snowflake Cutover

Obtain privileges; deploy DDL, audit, `NEXT`, `CURRENT`, and optional history;
activate Snowflake publisher/repository; pass equivalence tests; change
configuration; then remove local storage.

## 15. Acceptance Criteria

1. One refresh command calculates and validates independently of publisher.
2. Local publication creates typed, immutable, checksummed Parquet.
3. The pointer changes only after verification; failure retains prior data.
4. MCP never calculates features or owns the refresh connection.
5. MCP selects local or Snowflake repository through configuration.
6. Both repositories pass the same contract and parity tests.
7. Batch scoring returns all three probabilities and the selected class.
8. Missing, stale, corrupt, or incompatible data is never scored silently.
9. MCP hot-loads new local snapshots.
10. Model, schema, keywords, SQL, and publication lineage are traceable.
11. Credentials and sensitive data are not exposed.
12. The released model matches the production-safe calculation.
13. Persistent cutover changes storage configuration, not architecture.

## 16. Interim Limitations

- Different hosts need shared storage or snapshot distribution.
- Local inference lacks Snowflake table controls and query history.
- Filesystem availability and integrity become dependencies.
- Multiple MCP replicas need shared immutable snapshots.
- Local and Snowflake latency differ and require performance testing.
- Refresh still requires broad source read access.

These limitations are isolated to interim storage and do not alter the intended
persistent feature-store, MCP, or model design.
