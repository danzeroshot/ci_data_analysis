# Schedule Risk Model Training Pipeline Implementation Plan

## Execution Status

Last updated: 2026-07-19

| Workstream | Status | Current state |
| --- | --- | --- |
| Architecture and repository assessment | Complete | Architecture, proof-of-concept trainer, runtime loader, data snapshots, and tests reviewed |
| Host development environment | Complete | Python 3.8.10 host and pinned ML/report dependencies validated; Python 3.11 container remains blocked |
| Training configuration and artifact schemas | Complete | Strict Pydantic configuration, lineage hashes, stage status, and example policies implemented |
| Immutable label and training input snapshots | Complete | CSV and Snowflake SQL snapshot paths implemented; CSV snapshot created and marked development-only |
| Feature qualification | Complete | Static leakage controls and development-only screening implemented; 3,632 of 4,188 features qualified |
| Stable splits and manifests | Complete | SHA-256 holdout, stratified CV, temporal, and customer-isolation logic implemented |
| Random-forest tuning | Complete | Deterministic regularized forest search and weighted selection implemented |
| Metrics and reference profiles | Complete | Overall/per-class metrics, bootstrap intervals, fixed histograms, and prediction references implemented |
| Reporting and comparison | Complete | Self-contained HTML now distinguishes completed, limited, unavailable, not-run, and blocked results; external comparison with metric deltas and PSI is implemented |
| Filesystem release and rollback | Complete | Checksummed immutable bundles, atomic pointer, promotion, and rollback implemented |
| Runtime bundle integration | Complete | Runtime verifies checksums, status, class order, and parity; legacy loading remains a fallback |
| Host unit and integration validation | Complete | 12 tests pass, including temporal split coverage and end-to-end train, release, reload, rollback, and comparison |
| Docker image and Compose validation | Blocked | Docker client 26.1.3 is installed, but the daemon is inaccessible at /var/run/docker.sock |
| First controlled candidate run | Complete | Real-data development candidate released locally as schedule-rf-20260719T0347023-bebb4081 |
| Result-completeness validation run | Complete | Run schedule-rf-development-20260719T100419Z-ff12da52 produced completed temporal evaluation and the expanded self-contained report |
| Production release | Blocked | Requires client feature/target approval, release thresholds, and Docker validation |

Status values in this document are intended to be updated as implementation proceeds.

## Implementation Results

Host implementation completed on 2026-07-19.

Controlled run:

- Run ID: schedule-rf-development-20260719T034554Z-bebb4081
- Model version: schedule-rf-20260719T0347023-bebb4081
- Matched labeled projects: 3,469
- Development-only label source: approved historical CSV
- Candidate features: 4,188
- Qualified features: 3,632
- Rejected duplicate vectors: 438
- Rejected zero-variance fields: 117
- Rejected excessive-missingness fields: 1
- Locked holdout rows: 684
- Selected candidate: candidate-0002
- Locked-holdout macro F1: 0.6916
- Locked-holdout balanced accuracy: 0.7348
- Locked-holdout significant-delay recall: 0.8984
- Locked-holdout OVR macro ROC AUC: 0.9184
- Serialization parity: Passed
- Local promotion and rollback: Passed
- External comparison command: Passed
- Full host test suite: 11 passed

Result-completeness validation run:

- Run ID: schedule-rf-development-20260719T100419Z-ff12da52
- Model version: schedule-rf-20260719T100524Z-ff12da52
- Configured candidates completed: 2 of 2
- Normalized metric populations: development, CV out-of-fold, locked holdout, temporal holdout, Adams, CCD, Lincoln, and UDOT customer holdouts
- Customer coverage: 4 of 6 feature-snapshot customers evaluated; Amtrak has no matched labels and CLV has insufficient row/class support
- Temporal status: completed with 2,775 older training projects and 694 newer test projects at a 2024-03-21 boundary
- Temporal macro F1: 0.5744
- Temporal significant-delay recall: 0.8977
- Feature importance: exported for all 3,632 qualified predictors
- HTML evidence: 12 embedded charts, no relative plot dependencies, and no unresolved placeholders
- Full host test suite: 12 passed

The corrected feature snapshot contains PLANNEDSTARTDATE as split-only metadata.
It is used to define temporal validation populations and is not added to the
model feature schema.

The released candidate remains development-only because its labels came from the
historical CSV, numeric release thresholds are not approved, client feature and
target sign-off is pending, and Docker/Python 3.11 validation has not run.

## 1. Purpose

This plan implements schedule_risk_model_training_pipeline_architecture.md as a
production-oriented offline training component for the Schedule Risk Agent. It
is complementary to schedule_risk_agent_architecture.md and the existing runtime
implementation.

The immediate objective is to complete and validate the pipeline on the host
without Docker. Docker is a packaging and runtime-parity gate, not a dependency
for implementing data contracts, tuning, metrics, reports, or filesystem
releases.

The finished pipeline will:

1. Load immutable feature and label snapshots.
2. Qualify an approved beginning-only feature set.
3. Create deterministic development, holdout, temporal, and customer splits.
4. Tune only a regularized random forest.
5. Select a candidate with configurable emphasis on significant-delay recall.
6. Produce detailed metrics and fixed training reference profiles.
7. Serialize the exact model evaluated on the locked holdout.
8. Publish a versioned, checksummed filesystem bundle.
9. Allow later test data to be compared against the captured training state.
10. Supply the existing MCP runtime with a verified read-only model bundle.

## 2. Current Baseline

### 2.1 Existing Code

The current repository contains:

| Component | Current behavior | Required change |
| --- | --- | --- |
| schedule_risk_agent/train.py | Selects features, creates one random split, fits one fixed forest, and overwrites model files | Replace as the production entry point with a staged pipeline package; retain only as a temporary compatibility wrapper |
| schedule_risk_agent/evaluate.py | Reads model card and metrics JSON and prints them | Replace with external-data evaluation and reference comparison |
| schedule_risk_agent/model_runtime.py | Loads one joblib file, one model card, and one schema | Add verified bundle loading and retain a documented development fallback |
| schedule_risk_agent/config.py | Configures runtime and feature refresh | Keep runtime settings separate from immutable training-run configuration |
| tests/test_schedule_risk_agent.py | Covers feature selection, local feature retrieval, and probability response | Add a dedicated training test suite |
| schedule_risk_training_extract.sql | Documents a feature/label join but contains table placeholders | Split into executable label extraction and local immutable snapshot assembly |
| requirements.lock | Pins the runtime and core ML dependencies | Add only training/report dependencies that are actually required |

The current train.py must not be incrementally expanded into one large script.
The production implementation should be a package with independently testable
stages.

### 2.2 Available Data

A usable local feature snapshot already exists:

- Build ID: schedule-features-20260718T140401Z-fa9eac7d
- Rows: 5,762 projects
- Columns: 4,205
- Customers: Adams, Amtrak, CCD, CLV, Lincoln, and UDOT
- Format: Parquet
- Published feature schema version: schedule-project-features-v1
- Approved keyword version: approved-keywords-2026-06-11
- Current ordered model feature count: 4,188

The historical approved-keyword CSV contains 3,469 non-null PercentDelayed
labels and can bootstrap host development. It is not, by itself, sufficient
lineage for a production release because its extraction and label snapshot are
not yet represented as immutable artifacts.

### 2.3 Host and Container Runtimes

Verified host environment:

- Python 3.8.10
- NumPy 1.24.4
- pandas 2.0.3
- PyArrow 17.0.0
- scikit-learn 1.3.2
- joblib 1.4.2

The intended container uses Python 3.11. Work completed on Python 3.8 is
provisional until the same tests and serialization checks pass under Python 3.11
and in the final image.

### 2.4 Current Constraints

- Docker build, Compose, non-root filesystem, and container smoke tests cannot
  run because the installed Docker client cannot connect to the daemon.
- Persistent Snowflake feature-table privileges are unavailable.
- The local immutable Parquet feature snapshot is therefore the current training
  feature source.
- The label extraction SQL and Snowflake snapshot command are implemented, but a production-eligible Snowflake label snapshot has not been generated.
- Numeric release thresholds and the default significant-delay weight are not
  approved.
- Client approval of beginning-available fields and the retrospective target
  remains outstanding.

## 3. Implementation Decisions

### 3.1 Model Family

Tune only sklearn.ensemble.RandomForestClassifier.

Regularization controls include:

- max_depth;
- min_samples_leaf;
- min_samples_split;
- max_features;
- max_samples;
- class_weight; and
- a bounded tree count.

No alternate model family enters candidate selection in this implementation.

### 3.2 Selection Objective

Every run must explicitly provide significant_delay_weight, abbreviated w:

    selection_score =
        (1 - w) * macro_f1
        + w * recall_significant_delay

Valid range is 0.0 through 1.0.

- At 0.0, candidate selection uses macro F1 only.
- At 1.0, candidate selection uses significant-delay recall only.
- The value changes selection, not labels or reported metrics.
- The configuration has no silent production default.
- Example and development configurations may use a documented value, but release
  requires the value to be explicit in the immutable run configuration.

Minimum macro F1, precision, balanced accuracy, and overfitting guardrails remain
independent of this selection score.

### 3.3 Exact Evaluation Baseline

The release candidate is fit on the full development population after
cross-validation chooses parameters. That exact fitted object is evaluated once
on the locked hash holdout, serialized, and promoted without refitting on the
holdout.

This is required so later test performance is compared with metrics from the
same model object, not a similar model retrained on additional rows.

### 3.4 Customer Identity

CustomerName is retained for:

- project-key uniqueness;
- deterministic split hashing;
- subgroup reporting; and
- leave-one-customer-out testing.

It is never included in the predictor matrix.

### 3.5 Docker Independence

All pipeline modules, tests, local releases, and reports must run directly from
the repository. Docker-specific work is deferred to a separate phase and cannot
be used as an excuse to weaken host test coverage or artifact contracts.

## 4. Target Package and File Layout

Create the following package:

    schedule_risk_agent/
      training_pipeline/
        __init__.py
        __main__.py
        cli.py
        configuration.py
        contracts.py
        lineage.py
        snapshots.py
        feature_qualification.py
        splits.py
        random_forest.py
        metrics.py
        profiles.py
        reporting.py
        serialization.py
        release.py
        stages.py

Add configuration and policy files:

    config/
      schedule_training_run.example.json
      schedule_candidate_features.json
      schedule_release_policy.example.json

Add SQL and data preparation support:

    schedule_risk_label_calculation.sql
    schedule_risk_training_extract.sql
    scripts/
      create_schedule_label_snapshot.py
      create_schedule_training_manifest.py

Add dedicated tests:

    tests/training/
      conftest.py
      test_configuration.py
      test_snapshots.py
      test_feature_qualification.py
      test_splits.py
      test_selection_objective.py
      test_random_forest_search.py
      test_metrics.py
      test_profiles.py
      test_serialization.py
      test_release.py
      test_external_evaluation.py
      test_end_to_end_training.py

The module boundaries can be combined only when implementation shows that two
modules have no independent contract. The command surface and output contracts
must remain unchanged.

## 5. Configuration and Contract Foundation

### 5.1 Immutable Run Configuration

Implement a Pydantic model for the run configuration with these required
sections:

- run identity and random seed;
- immutable feature snapshot path and manifest;
- immutable label snapshot path and manifest;
- target definition version;
- candidate feature policy;
- significant-delay selection weight;
- random-forest search space;
- split policy;
- release policy path;
- artifact output root;
- resource controls; and
- report settings.

Validation requirements:

- no unknown top-level keys unless schema version explicitly permits them;
- significant-delay weight between 0 and 1;
- positive folds and candidate count;
- valid fractions whose meanings are explicit;
- all referenced files exist before a run starts;
- feature and label manifests include SHA-256 hashes;
- output root is not nested inside an immutable input snapshot;
- random seed is always explicit; and
- target and feature schema versions are nonempty.

Write the normalized configuration to run_config.json before reading model data.
Calculate and store its SHA-256 hash. Resuming a run with a different hash must
fail.

### 5.2 Stable Contract Objects

Implement typed contracts for:

- InputManifest;
- LabelManifest;
- FeatureQualificationRecord;
- SplitAssignment;
- CandidateParameters;
- MetricRecord;
- ReleaseGateResult;
- ArtifactManifest;
- RunStatus; and
- EvaluationComparison.

Every persisted JSON object includes a schema version. Parquet outputs carry the
schema version in their companion manifest.

### 5.3 Stage Status

Each stage writes stage_status.json containing:

- stage name;
- pending, running, succeeded, failed, or skipped status;
- start and finish timestamps;
- input hashes;
- output hashes;
- row and column counts;
- warning and error codes; and
- exception summary without secrets.

A stage is reusable only when its input hashes and configuration subset match.
Partial outputs are written to a staging path and removed or ignored on resume.

### 5.4 Concrete Acceptance Tests

- Invalid weighting values fail before any data is loaded.
- A changed input file invalidates resume.
- Unknown configuration fields fail with their full JSON path.
- Run IDs are unique and filesystem-safe.
- Secrets and Snowflake tokens cannot be serialized into run configuration or
  status output.

## 6. Immutable Training Data Pipeline

### 6.1 Feature Snapshot Input

Use the existing local feature snapshot format as the initial source. The loader
must:

1. Resolve the snapshot only through its manifest.
2. Verify COMPLETE, checksums, row count, and column count.
3. Verify FeatureSchemaVersion and KeywordManifestVersion.
4. Project only keys, split metadata, and candidate feature columns.
5. Normalize CustomerName and ProjectID only for matching, never as features.
6. Reject duplicate project keys.
7. Record Arrow and pandas data types.
8. Avoid loading raw text or unused columns.

Do not train against feature_snapshots/current.json directly. Resolve it once,
record the immutable build directory, and pin that directory in the training
manifest.

### 6.2 Label Calculation SQL

Replace the placeholder target branch with executable Snowflake SQL that emits
one row per CustomerName plus ProjectID with:

- target actual start and end;
- target planned start and end;
- actual and planned duration days;
- PercentDelayed;
- ScheduleRiskBin;
- target definition version;
- target calculation timestamp; and
- label exclusion reason when a target is invalid.

The SQL must preserve the existing agreed definition:

    actual_start = first valid posting date minus 30 days
    actual_end = last valid posting date
    PercentDelayed =
        100 * actual_duration_days / planned_duration_days - 100

Do not place these fields in the online feature calculation.

Explicit invalid-label reasons include:

- no valid posting date;
- missing planned start;
- missing planned end;
- planned duration less than or equal to zero;
- actual end before actual start;
- duplicate project key; and
- non-finite calculated percentage.

Outliers are reported and retained unless a separate, versioned target policy
excludes them.

### 6.3 Label Snapshot Writer

create_schedule_label_snapshot.py will support two sources:

1. Snowflake query source for the controlled pipeline.
2. Legacy approved-keyword CSV source for bootstrap development only.

Both write:

    training_snapshots/labels/<label-build-id>/
      labels.parquet
      exclusions.parquet
      manifest.json
      profile.json
      checksums.sha256
      COMPLETE

The manifest records source type, SQL or CSV hash, target version, row counts,
class counts, exclusion counts, customer coverage, timestamps, and package
versions.

A legacy-source manifest is marked development_only and cannot pass a production
release gate.

### 6.4 Join and Reconciliation

The training loader joins immutable features and valid labels by normalized
CustomerName plus ProjectID. It writes reconciliation.json with:

- feature rows;
- label rows;
- matched rows;
- feature-only rows;
- label-only rows;
- duplicate/conflicting keys;
- match rates overall and by customer;
- target class counts after join; and
- exclusion reasons.

Release policy sets minimum match-rate and class-support thresholds. No unmatched
row is silently discarded.

### 6.5 Temporal Integrity Caveat

The snapshot contains fields believed to be available at project beginning, but
it was calculated retrospectively. A field can still leak future information if
the underlying source value is overwritten during project execution.

The candidate manifest must therefore distinguish:

- available at beginning;
- historically immutable;
- currently assumed immutable;
- mutable without as-of history;
- rejected.

A production release is blocked until the client approves the released fields or
an as-of reconstruction is implemented.

## 7. Candidate Feature Manifest and Qualification

### 7.1 Manifest Generation

Generate schedule_candidate_features.json from:

- models/schedule_risk_feature_schema.json;
- project_feature_non_keyword_field_dictionary_2026-06-10.csv;
- the approved keyword review artifacts;
- schedule_risk_feature_calculation.sql; and
- explicit manual overrides.

Each feature entry contains:

- exact feature name;
- feature family;
- source field or SQL expression;
- expected numeric type;
- beginning-availability disposition;
- approval status;
- inference-schema presence;
- keyword manifest version where applicable;
- allowed missingness policy;
- impossible-value checks; and
- notes.

The generator produces a review diff whenever source inputs change. It must not
automatically approve a new column.

### 7.2 Mandatory Static Exclusions

Reject before examining outcome values:

- CustomerName;
- project identifiers;
- project names, codes, descriptions, and raw text;
- PercentDelayed and ScheduleRiskBin;
- all TARGET-prefixed columns;
- payment, posting, and change-order fields;
- current status and other outcome-adjacent workflow fields;
- fields marked No or Maybe unless later explicitly approved;
- unsupported dates, objects, arrays, and free text; and
- columns missing from the online inference schema.

### 7.3 Development-Only Statistical Qualification

After the locked holdout assignment is created, calculate qualification
statistics on development rows only:

- missing count and rate;
- finite count;
- unique count;
- zero variance;
- minimum and maximum;
- mean and standard deviation;
- selected quantiles;
- zero proportion;
- duplicate feature vectors; and
- type compatibility.

Do not use target correlation, feature importance, or holdout values to qualify
features in version 1.

Write feature_qualification.parquet and CSV for all accepted and rejected
features. Rejection reasons are stable codes plus human-readable detail.

### 7.4 Matrix Construction

- Preserve manifest order.
- Convert accepted numeric features to float32 where safe.
- Replace positive and negative infinity with null.
- Fit median imputation only on each training population.
- Optionally add missingness indicators through the fitted sklearn pipeline.
- Do not scale features because the model is tree based.
- Record transformed feature count and imputer statistics.
- Fail if a required feature disappears or is duplicated.

### 7.5 Qualification Tests

Use synthetic features to verify:

- target and identifier rejection;
- holdout values cannot affect missing-rate or variance decisions;
- a feature constant only in development is rejected;
- duplicate vectors are deterministic;
- feature order is stable;
- infinity becomes missing before fitting; and
- a newly seen column is rejected until manifested.

## 8. Deterministic Split Implementation

### 8.1 Locked Hash Holdout

Hash this canonical UTF-8 string with SHA-256:

    target_definition_version | normalized_customer | normalized_project_id | split_seed

Convert the first 64 hash bits to an unsigned integer and compare it with the
configured threshold. The default policy allocates 80 percent to development and
20 percent to locked holdout.

Persist every assignment before feature qualification. Verify:

- no key in both populations;
- assignments reproduce exactly across row order and process restarts;
- all classes have sufficient support; and
- per-customer allocation is reported.

If support is inadequate, the run fails. It does not inspect model performance
and move individual projects between splits.

### 8.2 Cross-Validation

Within development:

- use deterministic StratifiedKFold;
- default to five folds;
- save fold assignments;
- validate class support in every validation fold; and
- lower fold count only through an explicit small-sample policy.

All preprocessing is fit independently inside each fold.

### 8.3 Temporal Stress Test

Using planned start date:

- sort with a documented null-date policy;
- fit a separate model on the oldest 80 percent;
- evaluate on the newest 20 percent;
- preserve missing/invalid date rows in a reported excluded group; and
- record date boundaries and class support.

This stress test does not select hyperparameters.

### 8.4 Customer-Isolation Stress Tests

For each customer with configured minimum support:

- fit a separate model on all other customers;
- evaluate on that customer;
- retain undefined metrics with reason codes; and
- report excluded customers and reasons.

Customer identity is never included in the matrix.

### 8.5 Split Artifacts

Write:

- split_assignments.parquet;
- split_summary.json;
- class_support_by_split.csv;
- temporal_split_summary.json; and
- customer_holdout_summary.json.

## 9. Random-Forest Search

### 9.1 Search Engine

Implement an explicit deterministic search loop using ParameterSampler and
StratifiedKFold rather than relying on an opaque refit inside
RandomizedSearchCV. This provides direct control over:

- fold-level preprocessing;
- warnings and failures;
- out-of-fold predictions;
- checkpointing;
- resource limits;
- weighted selection; and
- tie breaking.

For each candidate:

1. Validate parameters against bounded policy.
2. Fit one pipeline per development fold.
3. Save fold predictions and probabilities.
4. Calculate fold and aggregate metrics.
5. Calculate train metrics for overfitting gaps.
6. Record fit and score durations.
7. Record warnings and failure reasons.
8. Write a completed candidate checkpoint.
9. Rank only after all valid candidates finish.

### 9.2 Search Space

Initial allowed values follow the architecture:

| Parameter | Allowed values |
| --- | --- |
| n_estimators | 250, 500, 800 |
| max_depth | 6, 8, 12, 16, null |
| min_samples_leaf | 5, 10, 15, 25 |
| min_samples_split | 10, 20, 40 |
| max_features | sqrt, 0.05, 0.10, 0.15 |
| max_samples | 0.70, 0.85, 1.00 |
| class_weight | balanced, balanced_subsample, approved custom mappings |
| criterion | gini, entropy, log_loss |

The run configuration controls sampled candidate count. It cannot expand values
outside release policy without a policy version change.

### 9.3 Resource Controls Without Docker

Add:

- model n_jobs setting;
- maximum concurrent candidates, initially one;
- optional BLAS thread limits;
- checkpoint after every candidate;
- elapsed-time and peak-memory telemetry;
- dry-run mode;
- maximum-candidate failure count; and
- clean interruption handling.

Do not combine outer candidate parallelism with unrestricted forest parallelism.

Before the full search:

1. Run a two-candidate, two-fold fixture test.
2. Run a five-candidate, three-fold benchmark on the real development data.
3. Record wall time, memory, and artifact growth.
4. Set the full candidate count and n_jobs from evidence.
5. Preserve the architecture default of 80 as a configurable target, not an
   assumption that the host can complete it economically.

### 9.4 Selection and Tie Breaking

Rank by:

1. Higher weighted selection score.
2. Higher macro F1.
3. Higher significant-delay recall.
4. Lower train-to-validation macro-F1 gap.
5. Shallower max depth.
6. Larger minimum leaf size.
7. Lower fit time.
8. Stable lexical parameter JSON.

Save the full ranked search table, not just the winner.

## 10. Metrics Implementation

### 10.1 Long-Form Metric Store

metrics_long.parquet contains one row per metric context with:

- run, model, data, target, and schema versions;
- candidate and selected flag;
- evaluation population;
- split family and fold;
- held-out customer when relevant;
- class label when relevant;
- support;
- metric name;
- value;
- undefined reason;
- random seed; and
- evaluation timestamp.

Undefined metrics remain rows with null value and a reason such as
class_absent_in_actual, class_absent_in_predictions, insufficient_support, or
numeric_failure.

### 10.2 Overall Metrics

Implement and test:

- accuracy;
- balanced accuracy;
- macro, weighted, and micro precision;
- macro, weighted, and micro recall;
- macro, weighted, and micro F1;
- selection score;
- multiclass log loss;
- one-vs-rest and one-vs-one ROC AUC;
- multiclass Brier score;
- expected and maximum calibration error;
- row count; and
- fit and inference timing.

Use explicit labels 0, 1, and 2 so missing predicted classes do not change array
shape.

### 10.3 Per-Class Metrics

For every class write:

- support and prevalence;
- predicted count and proportion;
- TP, FP, TN, and FN;
- precision, recall, specificity, and F1;
- false-negative and false-positive rates;
- one-vs-rest ROC AUC;
- average precision;
- Brier score; and
- calibration intercept and slope where support permits.

Significant-delay misses must be exposed directly as count, rate, and recall.

### 10.4 Confidence Intervals

For the locked holdout:

- use deterministic stratified bootstrap;
- default to 1,000 valid iterations;
- preserve class support where possible;
- report estimate and 95 percent interval;
- record attempted and valid iterations; and
- support paired bootstrap for candidate-versus-incumbent comparisons.

Provide a lower iteration count for unit tests and dry runs.

### 10.5 Train-to-Test Gaps

Calculate absolute and relative gaps for:

- macro F1;
- selection score;
- balanced accuracy;
- significant-delay precision and recall;
- log loss; and
- calibration error.

Handle zero denominators explicitly. Never show in-sample metrics without
cross-validation and holdout metrics beside them.

## 11. Final Candidate Evaluation and Reference State

### 11.1 Candidate Fit

After CV selection:

1. Recreate a fresh pipeline from selected parameters.
2. Fit it on all development rows only.
3. Evaluate it once on the locked holdout.
4. Run temporal and customer stress evaluations using separately fitted models.
5. Do not alter parameters based on locked or stress-test results.
6. Apply release gates after all results exist.

### 11.2 Feature Reference Profile

For every accepted feature capture from the release-candidate development data:

- type;
- missing and finite rates;
- minimum, maximum, mean, and standard deviation;
- 1st, 5th, 25th, 50th, 75th, 95th, and 99th percentiles;
- fixed histogram edges and counts;
- zero proportion; and
- monitoring outlier bounds.

Use robust deterministic bin construction. Constant and nearly constant features
must have valid, documented bins.

### 11.3 Prediction Reference Profile

Capture for development CV out-of-fold and locked holdout:

- actual class distribution;
- predicted class distribution;
- probability quantiles by actual and predicted class;
- prediction-confidence quantiles;
- top-two probability-margin quantiles;
- confusion matrices; and
- complete reference metrics.

### 11.4 External Test Comparison

Replace evaluate.py with a compatibility wrapper around:

    python -m schedule_risk_agent.training_pipeline evaluate
        --bundle <release>
        --features <feature-snapshot>
        --labels <label-snapshot>
        --output <evaluation-directory>

The comparison writes:

- current metrics;
- matching locked-holdout reference metrics;
- absolute and relative metric deltas;
- confidence intervals;
- fixed-bin feature PSI;
- missing-rate deltas;
- class and prediction distribution deltas;
- schema compatibility;
- pass, warn, or fail dispositions; and
- a readable HTML report.

The evaluator never recalculates training histogram boundaries from test data.

## 12. Reporting

Generate report.html with embedded or relative local graphics:

- executive metric summary;
- class distribution;
- normalized and count confusion matrices;
- one-vs-rest ROC and precision-recall curves;
- per-class precision, recall, and F1;
- calibration curves;
- CV candidate ranking;
- train/CV/holdout comparison;
- temporal performance;
- customer-isolation performance;
- feature and prediction drift;
- release-gate results; and
- limitations and lineage.

Also write machine-readable CSV, JSON, and Parquet behind every displayed table.

Use matplotlib with explicit noninteractive backend. Avoid notebook-only state.
The report generator must be callable in tests with a tiny fixture.

## 13. Serialization and Filesystem Release

### 13.1 Run Directory

Each run writes to:

    model_artifacts/schedule-risk/runs/<run-id>/

Follow the artifact layout in the architecture. Large controlled artifacts that
contain project identifiers remain outside the candidate bundle mounted by
inference.

### 13.2 Candidate Bundle

The candidate bundle contains at minimum:

- schedule_risk_model.joblib;
- schedule_risk_feature_schema.json;
- schedule_risk_model_card.json;
- schedule_risk_training_metrics.json;
- selected_hyperparameters.json;
- reference_metrics.json;
- feature_reference_profile.parquet;
- prediction_reference_profile.json;
- histogram_definitions.json;
- parity_input.parquet;
- parity_expected_predictions.json;
- requirements.lock;
- artifact_manifest.json; and
- checksums.sha256.

The full sklearn Pipeline is serialized. A parameter dictionary is never treated
as the model.

### 13.3 Verification

Before promotion:

1. Reload the model in a clean process.
2. Verify every checksum.
3. Verify package and Python compatibility declarations.
4. Verify ordered feature names and types.
5. Verify classifier classes are exactly 0, 1, and 2.
6. Re-score the parity sample.
7. Require probabilities and predictions to match configured tolerances.
8. Run a ModelRuntime smoke test against the bundle.
9. Confirm prohibited training artifacts are absent.

### 13.4 Release Gates

Hard gates implemented immediately:

- immutable inputs verified;
- no split overlap;
- all required classes present;
- no prohibited feature;
- inference schema parity;
- successful serialization round trip;
- successful parity prediction;
- all required artifacts and hashes present; and
- non-production input cannot be marked production.

Configurable numeric gates:

- minimum macro F1;
- minimum significant-delay recall;
- minimum balanced accuracy;
- maximum holdout calibration error;
- maximum train-to-holdout macro-F1 gap;
- minimum subgroup support; and
- allowable regression from incumbent.

Until numeric thresholds are approved, the pipeline can publish a candidate
release with development status but cannot mark it production_approved.

### 13.5 Atomic Promotion and Rollback

Promotion creates:

    model_artifacts/schedule-risk/releases/<model-version>/

The release directory is immutable. After verification, atomically replace
current.json with a pointer containing model version, relative release path,
manifest hash, and promotion metadata.

Rollback validates the selected prior release and atomically changes current.json.
It never edits or deletes the failed release.

## 14. Runtime Integration

Update ModelRuntime to prefer MODEL_BUNDLE_PATH or a resolved release pointer.

At startup it must:

- validate bundle checksums;
- load schema and model card;
- enforce release status policy;
- verify Python and sklearn compatibility;
- verify class mapping;
- run the parity sample; and
- expose loaded model and schema versions in readiness metadata.

Keep MODEL_PATH, FEATURE_SCHEMA_PATH, and MODEL_CARD_PATH only as a documented
development compatibility mode during migration. Production configuration must
use one bundle root to prevent mixed-version files.

Update schedule_risk_agent/README.md with training, evaluation, release,
rollback, and host-development commands.

## 15. CLI and Stage Orchestration

Implement these commands:

    python -m schedule_risk_agent.training_pipeline qualify --config <run.json>
    python -m schedule_risk_agent.training_pipeline tune --config <run.json>
    python -m schedule_risk_agent.training_pipeline evaluate --run-id <id>
    python -m schedule_risk_agent.training_pipeline release --run-id <id>
    python -m schedule_risk_agent.training_pipeline rollback --model-version <version>
    python -m schedule_risk_agent.training_pipeline run --config <run.json>
    python -m schedule_risk_agent.training_pipeline compare
        --bundle <release> --features <snapshot> --labels <snapshot>

Command behavior:

- run executes all non-promotion stages and preserves intermediates;
- release is explicit;
- qualify cannot tune;
- tune resumes qualification only when hashes match;
- evaluate cannot modify candidate selection;
- compare writes to a new evaluation directory;
- all commands support machine-readable JSON result output; and
- nonzero exit codes distinguish configuration, data, model, gate, and system
  failures.

## 16. Host-First Test Plan

### 16.1 Unit Tests

Implement the architecture tests for:

- configuration validation;
- hashing and lineage;
- feature exclusions;
- development-only screening;
- split determinism and disjointness;
- weighted score at 0, intermediate values, and 1;
- tie-break determinism;
- all metric formulas;
- absent-class reason handling;
- bootstrap reproducibility;
- fixed histogram and PSI calculations;
- artifact checksums;
- atomic pointer replacement; and
- release policy evaluation.

### 16.2 Integration Fixtures

Create a deterministic synthetic three-class dataset containing:

- all three classes;
- missing values;
- an infinite value;
- one prohibited target feature;
- one constant feature;
- one duplicate feature;
- customer and date groups; and
- a known significant-delay signal.

Use it for a complete qualify, tune, evaluate, serialize, release, reload, and
compare cycle in temporary directories.

### 16.3 Real-Data Host Validation

Run in this order:

1. Bootstrap a development label snapshot from the approved-keyword CSV.
2. Join it to the existing immutable feature snapshot.
3. Run qualification only and review all accepted/rejected counts.
4. Run the five-candidate benchmark.
5. Review resource usage and select a search budget.
6. Run a controlled candidate search.
7. Generate all holdout and stress-test outputs.
8. Verify bundle load through ModelRuntime.
9. Evaluate the same bundle against a copied fixture snapshot and verify zero
   unexpected metric drift.
10. Record that the result is development_only because labels came from CSV.

### 16.4 Commands on the Current Host

Use the pinned host interpreter initially:

    python3 -m pytest -q tests/training
    python3 -m schedule_risk_agent.training_pipeline run         --config config/schedule_training_run.development.json
    python3 -m schedule_risk_agent.training_pipeline release         --run-id <run-id>

Before implementation starts, capture pip freeze and platform metadata in the
run environment file. Do not install unpinned dependencies into the release
bundle.

## 17. Docker-Deferred Work

The following tasks remain blocked until Docker access is available:

1. Build docker/schedule-risk-agent.Dockerfile.
2. Confirm requirements install under Python 3.11.
3. Run the full unit and small end-to-end suite in the image.
4. Verify joblib created on the selected training runtime loads in the inference
   runtime.
5. Exercise the training command with writable artifact and snapshot mounts.
6. Exercise the MCP command with a read-only promoted bundle.
7. Verify non-root UID permissions and atomic filesystem operations.
8. Verify Compose volume ownership and restart behavior.
9. Verify CPU and memory controls.
10. Verify image contains training code but exposes no training MCP tool.
11. Run readiness and scoring smoke tests.
12. Record image digest and package inventory in the release candidate.

Until these pass:

- candidate bundles are host_validated only;
- promotion to production_approved is prohibited;
- the current development model remains clearly labeled as proof of concept; and
- no claim of Python 3.11 serialization compatibility is made.

When Docker becomes available, these are validation tasks, not a redesign.

## 18. Phased Execution Plan

### Phase 0: Baseline Preservation

Status: Complete

Tasks:

- Capture checksums of current models and schema.
- Preserve the current proof-of-concept model as a named legacy release.
- Capture host package and platform metadata.
- Add the execution-status table to implementation tracking.
- Ensure training artifacts and large snapshots are ignored or retained
  according to repository policy.

Exit criteria:

- Existing runtime behavior can be restored.
- No production pipeline work overwrites the current model files.

### Phase 1: Contracts and Skeleton

Status: Complete

Tasks:

- Create training_pipeline package and CLI.
- Implement configuration, contracts, lineage, stage status, and run directories.
- Add example configuration and policy files.
- Add unit tests for configuration and resume hashing.

Exit criteria:

- A dry run validates inputs and creates a reproducible empty run directory.
- Invalid configuration fails before model data is read.

### Phase 2: Immutable Inputs

Status: Complete

Tasks:

- Implement label calculation SQL.
- Implement CSV bootstrap and Snowflake label snapshot writers.
- Implement feature snapshot verification.
- Implement key reconciliation and exclusion reporting.
- Add small immutable fixtures.

Exit criteria:

- Features and labels join reproducibly.
- Every excluded or unmatched row has a reason.
- A development-only label source cannot pass a production release gate.

### Phase 3: Feature Qualification and Splits

Status: Complete

Tasks:

- Generate and review candidate feature manifest.
- Implement static exclusions.
- Implement development-only statistical screening.
- Implement locked, CV, temporal, and customer split manifests.
- Add leakage and determinism tests.

Exit criteria:

- Ordered accepted schema is reproducible.
- Holdout data cannot affect feature acceptance.
- Split assignments reproduce byte-for-byte.

### Phase 4: Search and Core Metrics

Status: Complete

Tasks:

- Implement random-forest candidate generator and bounded policy.
- Implement explicit CV search loop and checkpoints.
- Implement weighted objective and tie breaks.
- Implement overall and per-class metric library.
- Implement train/CV gap reporting.
- Run fixture search tests.

Exit criteria:

- Interrupted search resumes from complete candidates.
- Selection is deterministic.
- Weight values 0 and 1 behave exactly as specified.

### Phase 5: Final Evaluation and References

Status: Complete

Tasks:

- Fit the development-only release candidate.
- Evaluate locked holdout exactly once.
- Run temporal and customer stress tests.
- Implement bootstrap intervals.
- Build feature and prediction reference profiles.
- Implement external test comparison.

Exit criteria:

- Serialized model exactly matches the evaluated object.
- Later test results can be compared against fixed training-time references.

### Phase 6: Reporting and Release

Status: Complete

Tasks:

- Generate detailed HTML and machine-readable reports.
- Implement release gates.
- Build and verify candidate bundle.
- Implement atomic promotion and rollback.
- Extend ModelRuntime to load and verify bundles.
- Update package documentation.

Exit criteria:

- A verified host_validated release can be promoted and rolled back locally.
- Mixed model/schema/card versions are rejected.
- Reports clearly distinguish CV, locked holdout, and stress-test results.

### Phase 7: Real-Data Controlled Run

Status: Complete (development benchmark; temporal view unavailable in the pre-fix snapshot)

Tasks:

- Create bootstrap label snapshot.
- Run qualification and review.
- Benchmark search.
- Run approved host search budget.
- Review metrics, overfitting, subgroups, and release gates.
- Archive the complete run and client-facing report.

Exit criteria:

- One reproducible development candidate exists with complete lineage.
- All expected metrics and comparison references are populated.
- Known data and approval limitations are explicit.

### Phase 8: Docker Validation

Status: Blocked

Tasks are those in Section 17.

Exit criteria:

- Python 3.11 image tests pass.
- Training and inference commands work with intended mounts and permissions.
- A container-validated bundle can be considered for production approval.

## 19. Acceptance Criteria

Implementation is complete when:

- all training inputs are immutable and checksummed;
- feature and label lineage is complete;
- feature qualification records every candidate and disposition;
- CustomerName and all leakage fields are excluded from predictors;
- splits are deterministic, disjoint, and persisted;
- only the regularized random-forest family is tuned;
- significant-delay weighting is explicit and tested;
- selected hyperparameters and every candidate result are retained;
- CV, locked holdout, temporal, and customer metrics are complete;
- unexpected undefined metrics fail or warn according to policy;
- confidence intervals and calibration outputs are present;
- the exact evaluated pipeline is serialized;
- feature and prediction reference profiles support later comparison;
- release gates are machine-readable;
- promotion and rollback are atomic;
- the runtime rejects inconsistent or corrupted bundles;
- the full host test suite passes;
- the real-data development run is reproducible; and
- Docker-only criteria are visibly blocked rather than silently omitted.

Production readiness additionally requires:

- approved beginning-feature list;
- approved target definition and bins;
- approved significant-delay weight;
- approved numeric release thresholds;
- a non-development label snapshot;
- Python 3.11 and Docker validation; and
- explicit model promotion approval.

## 20. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Retrospectively calculated features contain overwritten future values | Optimistic validation | Client availability review, immutable-field flag, as-of reconstruction where needed |
| Only 3,469 currently labeled projects | Unstable class and subgroup metrics | Preserve holdout, confidence intervals, support thresholds, no silent subgroup removal |
| 4,188 predictors with relatively few rows | Overfitting and expensive search | Strong forest constraints, development-only qualification, CV gaps, bounded benchmark |
| Sparse keyword columns dominate compute | Slow fitting and large models | float32 matrix, projected reads, max_features limits, benchmark before full search |
| Significant-delay weighting drives trivial high-recall model | Poor precision and macro performance | Independent precision, macro F1, and balanced-accuracy gates |
| Hash holdout has weak class support | Invalid final evaluation | Pre-model support gate; versioned split policy change rather than manual row movement |
| Temporal or customer tests have one class | Undefined AUC or precision | Retain null metric with reason and emphasize supported metrics |
| Host Python differs from container Python | Serialization incompatibility | Host status only until Python 3.11 image round-trip passes |
| Legacy CSV label bootstrap lacks production lineage | Non-releasable candidate | Mark development_only and enforce hard release gate |
| Current model files are overwritten during development | Runtime regression | Preserve legacy bundle and use isolated run/release directories |
| Unapproved release thresholds | Ambiguous promotion | Permit candidate status only until policy is approved |
| Docker remains unavailable | Delayed deployment validation | Complete all host work and keep Docker as a bounded final gate |

## 21. Decisions Required Before Production Release

The implementation can proceed without these decisions, but production
promotion cannot:

1. Approve the target actual-start proxy and 0/25 percent bin boundaries.
2. Approve the beginning-available and historically stable feature list.
3. Set the production significant_delay_weight.
4. Set minimum macro F1, significant-delay recall, balanced accuracy, and
   precision.
5. Set maximum acceptable overfitting and calibration gaps.
6. Define minimum customer and class support for release.
7. Approve artifact retention, backup, and promotion authority.
8. Decide whether a calibrated probability stage is needed later.
9. Confirm Python 3.11 as the production training and inference runtime.
10. Restore Docker access for final packaging validation.

## 22. First Implementation Actions

Begin in this exact order:

1. Preserve the current proof-of-concept artifacts.
2. Create the training_pipeline package and configuration contracts.
3. Add the deterministic synthetic end-to-end fixture.
4. Implement immutable feature and bootstrap label snapshot readers.
5. Generate the initial candidate feature manifest.
6. Implement stable split assignments before any feature statistics.
7. Implement feature qualification.
8. Implement weighted metric functions.
9. Implement a two-candidate CV search.
10. Complete the fixture release and reload cycle.
11. Run qualification against the real local snapshot.
12. Benchmark real-data fitting before selecting the full search budget.

This order produces executable evidence early, protects the existing runtime,
and does not depend on Docker access.
