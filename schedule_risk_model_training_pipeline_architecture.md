# Schedule Risk Model Training Pipeline Architecture

## 1. Purpose

This document defines the offline training, tuning, evaluation, and release
pipeline for the Schedule Risk Agent. It complements
`schedule_risk_agent_architecture.md`, which remains authoritative for feature
publication and online MCP inference.

The training pipeline accepts an immutable labeled project dataset and a
candidate feature policy, identifies the usable model features, tunes one
regularized random-forest classifier, evaluates it without contaminating the
test populations, and publishes a versioned filesystem model bundle.

The released model server loads the fitted pipeline from that bundle. The
serialized artifact contains the complete fitted scikit-learn pipeline, not only
a dictionary of hyperparameters.

## 2. Objectives

The component must:

- reproduce every training run from immutable inputs and configuration;
- identify accepted and rejected features with explicit reasons;
- prevent target and temporal leakage;
- tune only the regularized random-forest model family;
- optimize primarily for macro F1 with configurable emphasis on avoiding missed
  `significant_delay` projects;
- preserve untouched test populations;
- capture detailed train, cross-validation, holdout, temporal, and customer
  metrics in a stable machine-readable schema;
- quantify differences between training and test performance;
- serialize the fitted preprocessing and classifier state;
- publish immutable, checksummed filesystem bundles;
- support atomic promotion and rollback;
- verify exact parity with the online feature schema and model loader; and
- provide reference distributions for later testing and drift analysis.

## 3. Boundaries

### In Scope

- Training-data and label ingestion.
- Candidate feature identification and screening.
- Stable split creation.
- Random-forest hyperparameter search.
- Model selection and release guardrails.
- Final refit and serialization.
- Detailed metrics, predictions, plots, and lineage.
- Versioned filesystem publication.
- Offline evaluation of a released bundle.
- Training/inference parity verification.

### Out of Scope

- Periodic feature refresh scheduling.
- Online MCP scoring.
- Production table creation and atomic feature publication.
- Automated retraining scheduling.
- Online learning.
- Alternative model families.
- Automatic business approval of a model.
- Final target-definition or feature-availability approval.

## 4. Relationship to Existing Architecture

`schedule_risk_agent_architecture.md` owns:

- Snowflake and local feature publication;
- the inference repository;
- MCP payloads and responses;
- model loading and readiness;
- online freshness and schema checks; and
- inference observability.

This document owns:

- training snapshots and labels;
- feature qualification;
- data splits;
- hyperparameter tuning;
- model evaluation and selection;
- release artifacts and reference metrics; and
- offline comparison against a released training state.

The contract between them is the released model bundle described in Section 15.

## 5. Target Contract

The model predicts:

| Class ID | Label | Definition |
| ---: | --- | --- |
| 0 | `no_delay` | `PercentDelayed <= 0` |
| 1 | `mild_delay` | `0 < PercentDelayed <= 25` |
| 2 | `significant_delay` | `PercentDelayed > 25` |

```text
PercentDelayed =
    100.0 * TargetActualDurationDays / TargetPlannedDurationDays - 100.0
```

The target remains versioned because its actual-start proxy and boundaries
require client approval. A target-definition change creates a new target version
and requires a complete retraining run. Runs with different target versions are
not directly comparable without an explicit reconciliation analysis.

Required target metadata:

- `target_definition_version`;
- source SQL hash;
- label extraction timestamp;
- number and percentage of unlabeled projects;
- exclusion counts by reason;
- class counts and class proportions; and
- minimum, maximum, and selected quantiles of `PercentDelayed`.

## 6. Training Inputs

A run receives one immutable JSON configuration:

```json
{
  "run_name": "schedule-rf-2026-07-candidate",
  "random_seed": 42,
  "training_snapshot": {
    "path": "/training/snapshots/schedule-features-<id>/features.parquet",
    "manifest_path": "/training/snapshots/schedule-features-<id>/manifest.json"
  },
  "labels": {
    "path": "/training/labels/schedule-labels-<id>.parquet",
    "target_definition_version": "schedule-delay-v1"
  },
  "feature_policy": {
    "manifest_path": "/config/schedule_candidate_features.json",
    "maximum_missing_rate": 0.95,
    "minimum_non_null_count": 50,
    "drop_zero_variance": true
  },
  "selection": {
    "significant_delay_weight": 0.35,
    "primary_metric": "weighted_macro_f1_significant_recall"
  },
  "tuning": {
    "strategy": "randomized_search",
    "iterations": 80,
    "cross_validation_folds": 5
  },
  "release_policy_path": "/config/schedule_release_policy.json",
  "output_root": "/model-artifacts/schedule-risk"
}
```

Inputs must include hashes or immutable IDs. A run must fail if an input changes
after its manifest was read.

## 7. Feature Identification

### 7.1 Candidate Sources

Candidate features come from:

1. An explicit versioned candidate-feature manifest.
2. The versioned online feature schema produced by the same feature calculation.
3. Approved keyword columns in the keyword manifest.
4. Approved non-keyword fields marked available at project beginning.

The pipeline does not accept arbitrary raw columns solely because they are
numeric.

### 7.2 Mandatory Exclusions

Reject:

- `CustomerName` as a predictor;
- project IDs, names, codes, descriptions, and raw text;
- `PercentDelayed`;
- every `TARGET*` field;
- payment and posting data;
- current status fields not known at project setup;
- change-order fields for model version 1;
- fields not approved as available at project beginning;
- unsupported object, list, or free-text values; and
- columns absent from the inference feature calculation.

### 7.3 Statistical Screening

Screen the remaining candidates for:

- entirely null values;
- configured missing-rate threshold;
- configured minimum non-null observations;
- zero variance;
- non-finite values;
- incompatible data types;
- duplicate column names;
- duplicate feature vectors;
- impossible values identified by the feature contract; and
- train/inference schema incompatibility.

Missing-rate and variance decisions are calculated on the development-training
population only. The locked final holdout must not influence feature acceptance.

Outcome-based univariate selection is disabled by default. If it is introduced
later, it must run only inside each cross-validation training fold.

### 7.4 Feature Qualification Output

Write `feature_qualification.parquet` and CSV with one row per candidate:

- feature name;
- family and source;
- expected type;
- beginning-availability disposition;
- accepted flag;
- rejection reason;
- development non-null count and missing rate;
- unique count;
- minimum, maximum, mean, standard deviation, and quantiles;
- non-finite count;
- zero-variance flag;
- duplicate-of feature, when applicable; and
- inference-schema presence.

The accepted ordered list becomes `schedule_risk_feature_schema.json`.

## 8. Split Architecture

### 8.1 Split Order

Split before tuning:

1. Remove rows without a valid target using documented reasons.
2. Create a locked final hash holdout.
3. Use only the remaining development population for feature screening and
   hyperparameter tuning.
4. Create cross-validation folds within the development population.
5. Preserve independent temporal and customer-holdout stress tests.

### 8.2 Stable Hash Holdout

Use a stable cryptographic hash of:

```text
target_version + CustomerName + ProjectID + split_seed
```

The default is 80% development and 20% final holdout. Store the hash algorithm,
seed, threshold, and assignments. The test set is not inspected during feature
qualification or model selection.

### 8.3 Cross-Validation

Use deterministic stratified K-fold cross-validation within development data.
Default: five folds. Save project assignments and class support per fold.

If a fold cannot represent all classes, the run fails unless a documented
small-sample policy explicitly permits fewer folds.

### 8.4 Temporal Test

Train on the oldest 80% by planned start date and test on the newest 20%.
Projects with missing or invalid dates are reported separately and excluded only
from this stress test.

### 8.5 Customer-Holdout Tests

For each customer with sufficient rows and class support:

- train on all other customers;
- test on the held-out customer; and
- report all standard and per-class metrics.

Tiny or single-class customer tests are retained in output with undefined metrics
and explicit reason codes. They are never silently dropped.

### 8.6 Split Manifest

`split_assignments.parquet` records project key, split family, fold, target,
planned date, and deterministic hash. This controlled artifact may contain
identifiers and is not mounted in the inference container.

## 9. Random-Forest Tuning

Only `RandomForestClassifier` is tuned in version 1. The serialized scikit-learn
`Pipeline` contains imputation and the classifier.

Candidate dimensions:

- `n_estimators`;
- `max_depth`;
- `min_samples_leaf`;
- `min_samples_split`;
- `max_features`;
- `max_samples`;
- `class_weight`;
- `criterion`; and
- optional missingness indicators in preprocessing.

Initial bounded ranges:

| Parameter | Candidate range |
| --- | --- |
| `n_estimators` | 250, 500, 800 |
| `max_depth` | 6, 8, 12, 16, null |
| `min_samples_leaf` | 5, 10, 15, 25 |
| `min_samples_split` | 10, 20, 40 |
| `max_features` | square root, 0.05, 0.10, 0.15 |
| `max_samples` | 0.70, 0.85, 1.00 |
| `class_weight` | balanced, balanced_subsample, approved custom weights |
| `criterion` | gini, entropy, log_loss |

Use deterministic randomized search by default. Record every sampled
configuration, fold result, fit time, score time, warning, and failure.

`class_weight` changes model fitting. It is separate from the model-selection
weight described next.

## 10. Weighted Selection Objective

Configuration:

```text
significant_delay_weight = w, where 0.0 <= w <= 1.0
```

For each cross-validation candidate:

```text
selection_score =
    (1 - w) * macro_f1
    + w * recall_significant_delay
```

Interpretation:

- `w = 0.0`: selection is ordinary macro F1.
- `w = 0.5`: macro F1 and significant-delay recall contribute equally.
- `w = 1.0`: selection is based entirely on significant-delay recall.

This weighting affects candidate selection only. It does not change metric
reporting, target labels, or class probability output.

Tie-breaking order:

1. Higher selection score.
2. Higher macro F1.
3. Higher significant-delay recall.
4. Lower train-to-validation macro-F1 gap.
5. Shallower tree depth.
6. Larger `min_samples_leaf`.
7. Lower fit time.
8. Stable lexical ordering of the parameter JSON.

Even at `w = 1.0`, configurable release guardrails can require minimum macro F1,
balanced accuracy, and precision so a model cannot win merely by labeling nearly
everything significant.

## 11. Metrics Architecture

### 11.1 Metric Context

Every metric row contains:

- metric schema version;
- run ID and model version;
- dataset snapshot and target versions;
- feature schema version;
- evaluation population;
- split family, fold, and held-out customer where applicable;
- candidate ID and selected-candidate flag;
- class support;
- metric name and value;
- undefined reason;
- random seed; and
- evaluation timestamp.

Populations include:

- development training, in-sample;
- cross-validation training fold;
- cross-validation validation fold;
- cross-validation out-of-fold aggregate;
- locked hash holdout;
- temporal holdout;
- each customer holdout;
- final refit training population; and
- subsequent external test datasets.

### 11.2 Overall Classification Metrics

Capture:

- accuracy;
- balanced accuracy;
- macro, weighted, and micro precision;
- macro, weighted, and micro recall;
- macro, weighted, and micro F1;
- selection score and configured `w`;
- multiclass log loss;
- multiclass one-vs-rest ROC AUC, macro and weighted;
- multiclass one-vs-one ROC AUC, macro and weighted;
- multiclass Brier score;
- expected calibration error;
- maximum calibration error;
- sample count; and
- prediction latency.

### 11.3 Per-Class Metrics

For every class capture:

- support and prevalence;
- predicted count and predicted proportion;
- true positives, false positives, true negatives, and false negatives;
- precision;
- recall/sensitivity;
- specificity;
- F1;
- false-negative and false-positive rates;
- one-vs-rest ROC AUC;
- average precision/PR AUC;
- Brier score; and
- calibration intercept and slope where statistically valid.

Missed significant delays are directly represented by:

```text
significant_delay_false_negative_count
significant_delay_false_negative_rate
significant_delay_recall
```

### 11.4 Confusion Matrices

Store count and normalized confusion matrices:

- normalized by actual class;
- normalized by predicted class; and
- normalized over all observations.

Write JSON/CSV plus client-readable PNG or HTML plots.

### 11.5 Training-versus-Test Gaps

For each metric calculate:

```text
absolute_gap = training_value - test_value
relative_gap = absolute_gap / abs(training_value)
```

Primary gap reporting:

- macro F1;
- selection score;
- balanced accuracy;
- significant-delay recall;
- significant-delay precision;
- log loss; and
- calibration error.

In-sample metrics are never presented without cross-validation and holdout
metrics beside them.

### 11.6 Confidence Intervals

Use deterministic stratified bootstrap resampling for locked holdout metrics.
Default: 1,000 iterations. Report estimate, lower/upper 95% intervals, valid
bootstrap count, and random seed.

For candidate-versus-incumbent comparison, use paired bootstrap samples over the
same projects.

### 11.7 Subgroup Metrics

Report, without using subgroup identity as model input:

- customer;
- planned-duration band;
- planned-value band;
- contract-item-count band;
- feature-missingness band; and
- class support.

Suppress or mark unstable subgroups below configurable sample/class thresholds.

### 11.8 Calibration and Threshold Metrics

Random-forest probabilities are not assumed calibrated. Capture reliability
curves, Brier scores, and calibration errors.

The model still predicts by maximum class probability unless a separately
approved decision policy is versioned. Threshold sweeps may be reported for
analysis but are not silently embedded in the fitted model.

## 12. Reference State for Later Testing

The release bundle must provide a training reference state that any subsequent
test run can compare against.

### Feature Reference Profile

For each accepted feature store:

- data type;
- non-null and missing counts/rates;
- finite count;
- mean, standard deviation, minimum, maximum;
- 1st, 5th, 25th, 50th, 75th, 95th, and 99th percentiles;
- fixed histogram bin edges and training counts;
- zero proportion;
- outlier bounds used by monitoring; and
- optional top values for low-cardinality numeric fields.

These fixed bins support PSI and distribution comparisons without recalculating
training boundaries from test data.

### Prediction Reference Profile

Store:

- true class distribution;
- predicted class distribution;
- probability quantiles by predicted and actual class;
- confidence/margin quantiles;
- confusion matrices; and
- all reference metrics.

### Comparison Output

`evaluate` writes:

- current test metric;
- matching training/CV/holdout reference metric;
- absolute and relative deltas;
- confidence intervals where available;
- pass/warn/fail disposition from comparison policy;
- feature missing-rate deltas;
- PSI or configured drift statistic;
- class-distribution delta; and
- model/schema compatibility result.

The comparison must identify which reference population is used. The default
client-facing comparison is locked training-time holdout versus the new test
population, not in-sample training performance.

## 13. Release Guardrails

Release policy is versioned and configurable. Candidate gates include:

- all required artifacts and hashes exist;
- feature and inference parity passes;
- no split overlap;
- all classes represented in development and locked holdout;
- minimum macro F1;
- minimum significant-delay recall;
- minimum balanced accuracy;
- maximum train-to-holdout macro-F1 gap;
- maximum calibration error when probabilities are presented;
- no prohibited feature;
- no unacceptable customer/subgroup collapse;
- no configured regression versus incumbent;
- successful serialization round trip; and
- successful model-server smoke test.

Thresholds are not hardcoded into training code. A failed gate produces a
complete rejected-candidate bundle and never changes the released pointer.

## 14. Final Fit and Locked Evaluation

After cross-validation selects hyperparameters:

1. Record the selected hyperparameters.
2. Fit one release-candidate pipeline on the complete development population only.
3. Evaluate that fitted pipeline once on the untouched locked hash holdout.
4. Record all locked holdout predictions, metrics, confidence intervals, and reference profiles.
5. Serialize and reload that exact evaluated pipeline without refitting it on holdout rows.
6. Verify predictions against a fixed parity sample.
7. Build and checksum the release bundle.

This default preserves an exact relationship between the serialized model and its locked-holdout baseline. Later tests can therefore compare against metrics from the same fitted object. Training a future release on development plus locked holdout requires a separately versioned policy and a new independent evaluation population; its prior holdout metrics cannot be represented as metrics of the refitted object.

Temporal and customer-isolation stress tests use separately fitted evaluation models with the selected hyperparameters. Their results characterize generalization risk but are not metrics of the serialized release candidate.

## 15. Filesystem Artifact Architecture

```text
/model-artifacts/schedule-risk/
  runs/
    <run-id>/
      run_config.json
      input_manifest.json
      environment.json
      data_profile.json
      feature_qualification.parquet
      feature_qualification.csv
      split_assignments.parquet
      hyperparameter_search_results.parquet
      hyperparameter_search_results.csv
      selected_hyperparameters.json
      metrics/
        metrics_long.parquet
        metrics_summary.json
        cv_fold_metrics.json
        train_metrics.json
        locked_holdout_metrics.json
        temporal_metrics.json
        customer_holdout_metrics.json
        confidence_intervals.json
        comparison_to_incumbent.json
      predictions/
        cv_out_of_fold.parquet
        locked_holdout.parquet
        temporal_holdout.parquet
        customer_holdouts.parquet
      reference/
        feature_reference_profile.parquet
        prediction_reference_profile.json
        histogram_definitions.json
      plots/
      report.html
      candidate/
        schedule_risk_model.joblib
        schedule_risk_feature_schema.json
        schedule_risk_model_card.json
        schedule_risk_training_metrics.json
        selected_hyperparameters.json
        reference_metrics.json
        feature_reference_profile.parquet
        prediction_reference_profile.json
        histogram_definitions.json
        requirements.lock
        checksums.sha256
      run_result.json
  releases/
    <model-version>/
      <immutable released bundle>
  current.json
```

Runs write to a staging directory. Promotion copies or hard-links the verified
candidate to an immutable release directory, then atomically replaces
`current.json`.

The inference container mounts only the selected release bundle read-only. It
does not mount split assignments, labels, or training predictions.

## 16. Serialization Contract

Use `joblib.dump` to serialize the complete fitted, trusted scikit-learn
`Pipeline`. The artifact is pickle-based and must never be loaded from an
untrusted source.

The pipeline contains:

- ordered feature input contract;
- fitted imputer statistics;
- optional fitted missingness indicators;
- fitted random-forest trees;
- class order; and
- scikit-learn preprocessing state.

Also write `selected_hyperparameters.json` for human inspection. Hyperparameters
alone are insufficient for inference.

The model loader verifies:

- bundle checksum;
- model and schema versions;
- Python, NumPy, and scikit-learn compatibility;
- ordered feature list;
- class mapping `[0, 1, 2]`;
- model card status;
- release approval state; and
- parity-sample predictions.

## 17. Model Card

The model card records:

- model version and release status;
- run ID;
- input snapshot, label, SQL, schema, and keyword versions;
- candidate and accepted feature counts;
- exclusion summary;
- target definition and class prevalence;
- `significant_delay_weight` and selection formula;
- selected hyperparameters;
- all headline CV, locked holdout, temporal, and customer metrics;
- confidence intervals;
- training-to-test gaps;
- incumbent comparison;
- release gate results;
- known limitations;
- package versions;
- training timestamp and duration; and
- approval identity/time when available.

## 18. Commands

Recommended command surface:

```text
python -m schedule_risk_agent.training_pipeline qualify --config <run.json>
python -m schedule_risk_agent.training_pipeline tune --config <run.json>
python -m schedule_risk_agent.training_pipeline evaluate --run-id <id>
python -m schedule_risk_agent.training_pipeline release --run-id <id>
python -m schedule_risk_agent.training_pipeline run --config <run.json>
python -m schedule_risk_agent.training_pipeline evaluate --bundle <release> --data <test>
```

`run` performs all stages but preserves each intermediate. `release` is a
separate explicit action unless automated promotion is later approved.

## 19. Failure and Resume Behavior

Each stage writes a status record:

- pending;
- running;
- succeeded;
- failed; or
- skipped because a prerequisite failed.

A run may resume only when input/config hashes match. Completed immutable stages
are reused; partial stage output is discarded and rerun.

Failures preserve diagnostics but never produce or promote a release.

## 20. Observability

Structured training events include:

- run ID and stage;
- input and configuration hashes;
- row/feature counts;
- candidate and fold IDs;
- fit/score duration;
- CPU and memory usage;
- warning/error codes;
- selected parameters and score;
- artifact paths and checksums; and
- release gate decisions.

No raw Snowflake tokens or unrestricted source text are logged.

## 21. Testing Strategy

### Unit Tests

- Feature exclusion and reason assignment.
- Missing/variance screening fit only on development data.
- Stable hash and fold determinism.
- Split disjointness.
- Weighted selection score at `w=0`, intermediate values, and `w=1`.
- Tie-breaking determinism.
- Metric and confidence-interval calculations.
- Undefined metric reason handling.
- Artifact hashing and atomic pointer updates.
- Configuration validation.

### Integration Tests

- Small complete training run.
- Resume after injected stage failure.
- Repeated run produces identical splits and selected parameters.
- Serialization/reload produces identical predictions.
- Released schema exactly matches inference extraction.
- Model server loads and scores a released bundle.
- Rejected candidate cannot change `current.json`.
- External test evaluation generates reference deltas.

### Leakage Tests

- Target columns are rejected.
- Holdout values cannot affect feature screening or tuning.
- Project keys never enter the matrix.
- Customer is excluded from the matrix.
- Fit-only transformations are not fit on validation/test rows.
- Temporal and customer holdouts remain evaluation-only.

## 22. Implementation Sequence

1. Define metric, artifact, configuration, and release-policy schemas.
2. Implement immutable input and lineage validation.
3. Implement feature qualification and reports.
4. Implement stable splits and manifests.
5. Implement weighted scorers and metric library.
6. Implement deterministic random-forest search.
7. Implement locked and stress-test evaluation.
8. Implement reference profiles and comparison reports.
9. Implement final refit and serialization.
10. Implement filesystem release and rollback.
11. Extend the runtime loader for bundle-level verification.
12. Add training/inference parity fixtures.
13. Add the complete test suite.
14. Run the first controlled production-safe training cycle.
15. Review metrics, weighting, guardrails, and release with stakeholders.

## 23. Remaining Decisions

Before the first production release, approve:

- default `significant_delay_weight`;
- randomized-search iteration and compute budget;
- minimum release metrics;
- allowable incumbent regression;
- minimum subgroup support;
- model-artifact retention;
- whether probability calibration becomes a later fitted stage;
- approval authority for promotion; and
- filesystem backup and disaster-recovery policy.

