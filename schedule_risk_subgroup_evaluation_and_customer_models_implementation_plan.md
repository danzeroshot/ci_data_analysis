# Schedule Risk Subgroup Evaluation and Customer Models Implementation Plan

## 1. Document Status

| Item | Status |
|---|---|
| Architecture decisions | Approved for implementation |
| Current-data eligibility preflight | Complete |
| Subgroup evaluation implementation | Not started |
| Customer-specific model implementation | Not started |
| Runtime dual-model response implementation | Not started |
| Production performance thresholds | Not approved |
| Subgroup evaluator and artifacts | Complete; validated on current run |
| Customer-specific training and artifacts | Complete; validated on current run |
| Dual-model runtime response | Implemented; focused runtime validation pending |
| Full automated regression coverage | Partial; focused tests pending |

This plan extends the schedule-risk training and runtime architecture. It defines five evaluation families, introduces optional customer-specific models alongside the mandatory all-customer model, and makes the four non-customer subgroup-family eligibility checks a production-promotion gate.

The plan does not add new subgroup families. It does not choose between the all-customer and customer-specific predictions. That choice belongs to a higher application layer.

## 2. Objectives

1. Evaluate the all-customer model consistently across customer, planned project duration, planned project value, contract item count, and qualified-predictor missingness.
2. Prove that each required non-customer subgroup family has enough examples of every delay class to support a meaningful evaluation.
3. Block promotion of a new all-customer production model when a required non-customer subgroup family is ineligible.
4. Train a customer-specific model whenever that customer's data supports one.
5. Reuse all-customer hyperparameters when a customer cannot support independent tuning but can still support fitting and evaluation.
6. Return both all-customer and customer-specific predictions at inference time, with explicit availability and lineage.
7. Preserve incumbent releases when any new candidate fails its applicable gate.

## 3. Scope and Non-Goals

### In scope

- Versioned subgroup definitions and support rules.
- Subgroup assignment, eligibility, metrics, charts, reports, and artifacts.
- A production gate requiring all four non-customer subgroup families to be eligible.
- Independently tuned or global-hyperparameter customer-specific random forests.
- Customer-specific artifact storage, release pointers, and runtime loading.
- MCP response changes needed to expose both model results.
- Unit, integration, report-schema, release, and runtime tests.
- Updates to architecture and operator documentation.

### Out of scope

- Additional subgroup families or interactions between subgroup families.
- Using subgroup identity, including `CUSTOMERNAME`, as a predictor.
- Automatically selecting which model result a caller should use.
- Different model families for particular customers.
- Separate subgroup performance thresholds in the first implementation.
- Resplitting data to make a customer or subgroup pass an eligibility gate.
- Online learning or runtime model retraining.

## 4. Terminology

| Term | Definition |
|---|---|
| All-customer model | Customer-agnostic model trained from every eligible development project. It is mandatory for production. |
| Customer-specific model | Separate customer-only model fitted on that customer's development projects. It is optional per customer. |
| Family | One partitioning dimension, such as planned duration. |
| Band | One mutually exclusive value range within a family. |
| Eligibility | Whether observations support evaluating a family or fitting/evaluating a customer model. |
| Performance gate | Approved metric thresholds a fitted model must satisfy before release. |
| Development population | Projects assigned to development by the existing stable hash split. |
| CV OOF population | One prediction per development project from a model that did not train on that project's CV fold. |
| Locked holdout | Existing deterministic 20% hash holdout, unused in tuning and model selection. |
| Tuning gate | Customer support required for independent customer-only hyperparameter tuning. |
| Absolute modeling floor | Lower customer support required to fit with frozen all-customer hyperparameters. |
| Ineligible family | A family where any required band or common validity criterion fails. No per-band performance evaluation runs for that family. |

## 5. Fixed Subgroup Definitions

All definitions must be stored in versioned configuration and copied into every run result. Boundaries are fixed before metrics are calculated and are never recomputed from each dataset.

### 5.1 Planned duration

Source: `PLANNEDDURATIONDAYS`.

| Band ID | Rule |
|---|---|
| `duration_lt_60` | `PLANNEDDURATIONDAYS < 60` |
| `duration_60_to_364` | `PLANNEDDURATIONDAYS >= 60 AND PLANNEDDURATIONDAYS < 365` |
| `duration_ge_365` | `PLANNEDDURATIONDAYS >= 365` |

Null, nonnumeric, nonfinite, or nonpositive duration fails assignment coverage. No hidden `unknown` band is added.

### 5.2 Planned project value

Source: `PROJECTPLANNEDVALUE`.

| Band ID | Rule |
|---|---|
| `value_le_0` | `PROJECTPLANNEDVALUE <= 0` |
| `value_0_to_1m` | `PROJECTPLANNEDVALUE > 0 AND PROJECTPLANNEDVALUE < 1000000` |
| `value_1m_to_10m` | `PROJECTPLANNEDVALUE >= 1000000 AND PROJECTPLANNEDVALUE < 10000000` |
| `value_ge_10m` | `PROJECTPLANNEDVALUE >= 10000000` |

Zero and negative values are retained as an explicit band because they are present in the current source and may represent source-system states encountered at runtime. Null, nonnumeric, or nonfinite values fail assignment coverage.

### 5.3 Contract item count

Source: `NUMCONTRACTITEMS`.

| Band ID | Rule |
|---|---|
| `items_1_to_19` | `NUMCONTRACTITEMS >= 1 AND NUMCONTRACTITEMS < 20` |
| `items_20_to_49` | `NUMCONTRACTITEMS >= 20 AND NUMCONTRACTITEMS < 50` |
| `items_ge_50` | `NUMCONTRACTITEMS >= 50` |

Null, nonnumeric, nonfinite, nonpositive, or materially nonintegral values fail assignment coverage. The earlier candidate split at 10 items is not used because its smallest band lacked sufficient mildly delayed holdout projects.

### 5.4 Qualified-predictor missingness

For each project:

    missing_predictor_rate =
        count(qualified predictor values that are missing) /
        count(qualified predictors)

The calculation uses the exact ordered list in `qualified_feature_schema.json`, after feature qualification and before imputation. A value is missing when the current `numeric_matrix` process would produce `NaN`: the source is null, cannot be converted by `pandas.to_numeric`, or is positive or negative infinity.

| Band ID | Rule |
|---|---|
| `missingness_zero` | `missing_predictor_rate == 0` |
| `missingness_nonzero` | `missing_predictor_rate > 0` |

The denominator and qualified-schema checksum are stored with each assignment. Failure to load the exact schema makes the family ineligible. A three-band split is not used because current data cannot support every class in its intermediate band.

### 5.5 Customer

Source: normalized `CUSTOMERNAME`, retained only as an evaluation and routing key. It is never included in either model's predictor matrix.

Customer work has three distinct parts:

1. Performance of the all-customer model within each customer's rows.
2. Leave-one-customer-out evaluation for unseen-customer generalization.
3. Training and evaluation of an optional customer-specific model.

Customer support failures do not block the all-customer release and do not affect any other customer's model.

## 6. Common Subgroup Eligibility Contract

### 6.1 Required support

Every band in a non-customer family must satisfy all conditions:

| Population | Minimum rows per band | Minimum rows per class per band |
|---|---:|---:|
| Development / pooled CV OOF | 50 | 4 |
| Locked holdout | 20 | 2 |

Class support is a common eligibility criterion, not a sixth subgroup family.

### 6.2 Family-level checks

| Check ID | Passing condition |
|---|---|
| `family_config_valid` | Boundaries are ordered, exhaustive over valid values, and nonoverlapping. |
| `source_field_present` | Every required source field or schema artifact is present. |
| `assignment_complete` | Every CV OOF and holdout row maps to exactly one band. |
| `band_count_exact` | Every configured band exists in both populations. |
| `prediction_join_complete` | Every assigned project has exactly one prediction and label in its population. |
| `cv_total_support` | Every band has at least 50 CV OOF rows. |
| `cv_class_support` | Every band has at least 4 CV OOF rows from each class. |
| `holdout_total_support` | Every band has at least 20 locked-holdout rows. |
| `holdout_class_support` | Every band has at least 2 locked-holdout rows from each class. |

Each result records expected value, actual value, status, failed bands, class counts, and an actionable reason.

### 6.3 Entire-family behavior

If any check fails:

- Mark the family `ineligible`.
- Do not calculate performance metrics for any band in that family.
- Do not render performance charts that could be mistaken for complete results.
- Render an eligibility table containing every band and failed criterion.
- Preserve assignment diagnostics for investigation.
- Continue evaluating other families.

This prevents a report from comparing only convenient bands while silently omitting an under-supported one.

## 7. Global Release Behavior

The required non-customer families are planned duration, planned value, contract item count, and qualified-predictor missingness. All four must be eligible before a new all-customer candidate can be promoted to production.

Add a machine-readable gate named `required_subgroup_family_eligibility` containing the required families, failed families, status, and detailed checks.

When the gate fails:

- The training run still completes.
- Candidate, diagnostic, eligibility, and report artifacts are retained.
- The candidate is classified as rejected for production.
- The production release pointer is not changed.
- The incumbent production model remains active.

Initially, subgroup performance is diagnostic. There is no separate numeric performance threshold per band until reviewed values are approved. Eligibility is still a hard gate because an unevaluable required family means model behavior is insufficiently characterized.

The existing global metric thresholds are currently null. Therefore, the current development run can generate complete artifacts but cannot be called production-approved until thresholds and production label lineage are approved.

## 8. Subgroup Metrics and Presentation

For each eligible family and band, calculate metrics from pooled CV out-of-fold predictions and locked-holdout predictions.

Required metrics:

- Row count and class counts.
- Observed and predicted class prevalence.
- Accuracy, macro F1, weighted F1, and balanced accuracy.
- Precision, recall, and F1 for each class.
- Significant-delay recall.
- Count and row-normalized confusion matrices.
- One-vs-rest ROC AUC and average precision where defined.
- Multiclass log loss and Brier score.
- Expected calibration error.

Every potentially undefined metric must be represented as `null` with an `unavailable_reason`. It must not disappear or become an unexplained `NaN`.

The report must show:

1. A gate matrix before subgroup performance.
2. Counts and class support for every configured band.
3. Overall-population values beside each band.
4. Absolute and relative differences from the overall population.
5. Confusion and class-recall charts for each eligible family.
6. Calibration and predicted-probability views where support permits.
7. A clear statement that subgroup performance is diagnostic.

Bootstrap confidence intervals use project-level resampling within each population and record attempted and valid iteration counts.

## 9. All-Customer Model

The existing all-customer path remains primary:

1. Preserve the deterministic project-key hash split.
2. Qualify features on development only.
3. Exclude `CUSTOMERNAME` and identifiers from predictors.
4. Tune the regularized random forest on development CV folds.
5. Fit the selected model on all development rows.
6. Evaluate once on locked holdout.
7. Generate subgroup and stress results after predictions are frozen.
8. Apply serialization, lineage, global metric, and required-family gates.

Every customer receives this model's prediction when it is available, including customers that cannot support their own model.

## 10. Customer Model Eligibility and Training

### 10.1 Independent-tuning gate

| Population | Minimum total | Minimum each class |
|---|---:|---:|
| Customer development rows | 50 | 4 |
| Customer locked holdout | 20 | 2 |

The customer CV fold count is reduced deterministically when necessary under the existing split rule, but never below two folds.

### 10.2 Absolute customer-modeling floor

If independent tuning is ineligible, a model may still be fitted with selected all-customer hyperparameters when:

| Population | Minimum total | Minimum each class |
|---|---:|---:|
| Customer development rows | 30 | 2 |
| Customer locked holdout | 10 | 2 |

This floor is fixed and is not adjusted to make a customer eligible.

### 10.3 Training disposition

| Condition | Action | Training mode |
|---|---|---|
| Tuning gate passes | Tune the same versioned RF search space with customer development data only. | `customer_tuned` |
| Tuning fails and absolute floor passes | Reuse all-customer selected hyperparameters and fit customer development data only. | `global_parameters_customer_fit` |
| Absolute floor fails | Do not fit or serialize. Report each failed criterion. | `unavailable` |

The existing locked split is reused. Rows are never moved to rescue eligibility.

### 10.4 Customer release rules

- Apply the same approved metric thresholds as the all-customer model.
- Apply serialization parity and artifact integrity independently.
- A failed customer gate rejects only that customer's candidate.
- Customer failure never blocks global or another customer.
- A customer model can be promoted only against a released parent all-customer version.
- A failed candidate leaves that customer's incumbent active.
- Until numeric thresholds are approved, customer artifacts remain development candidates.

### 10.5 Leave-one-customer-out diagnostic

Retain this as a separate test of transfer to an unseen customer. It is not a customer-specific model and must not be labeled as one.

Use the existing minimum of 20 test projects and 2 of every class. Report all customers, including `ineligible` and `no_labels`. Individual failures are diagnostic and do not block global promotion.

## 11. Runtime and MCP Contract

At startup, runtime must load and verify the mandatory all-customer release, then independently load zero or one active release per normalized customer. A customer-bundle failure is nonfatal to global inference.

For each `CustomerName + ProjectID`, return both result slots:

    {
      \"customer_name\": \"UDOT\",
      \"project_id\": \"12345\",
      \"models\": {
        \"all_customer\": {
          \"status\": \"available\",
          \"model_version\": \"...\",
          \"predicted_class\": 2,
          \"class_probabilities\": {
            \"no_delay\": 0.08,
            \"mild_delay\": 0.17,
            \"significant_delay\": 0.75
          }
        },
        \"customer_specific\": {
          \"status\": \"available\",
          \"model_version\": \"...\",
          \"parent_all_customer_model_version\": \"...\",
          \"training_mode\": \"customer_tuned\",
          \"predicted_class\": 2,
          \"class_probabilities\": {
            \"no_delay\": 0.04,
            \"mild_delay\": 0.11,
            \"significant_delay\": 0.85
          }
        }
      }
    }

When unavailable, the customer slot has `status=unavailable`, a stable reason code, failed eligibility criteria when applicable, and a statement that the all-customer result remains valid.

Do not include `selected_model` or `recommended_model`. In batch scoring, one customer-specific failure must not fail other projects. A missing or invalid all-customer release is a service-level fatal error.

The initial implementation requires exact global/customer feature-schema equality. Both bundles record target-definition, schema, and parent model versions.

## 12. Configuration Changes

Add strict versioned configuration equivalent to:

    {
      \"subgroup_evaluation\": {
        \"schema_version\": \"schedule-subgroups-v1\",
        \"required_for_global_production\": [
          \"planned_duration\",
          \"planned_value\",
          \"contract_item_count\",
          \"predictor_missingness\"
        ],
        \"development_minimum_rows_per_band\": 50,
        \"development_minimum_rows_per_class_per_band\": 4,
        \"holdout_minimum_rows_per_band\": 20,
        \"holdout_minimum_rows_per_class_per_band\": 2,
        \"duration_boundaries_days\": [60, 365],
        \"planned_value_boundaries\": [0, 1000000, 10000000],
        \"contract_item_boundaries\": [20, 50],
        \"missingness_mode\": \"zero_vs_nonzero\"
      },
      \"customer_models\": {
        \"enabled\": true,
        \"tuning_development_minimum_rows\": 50,
        \"tuning_development_minimum_rows_per_class\": 4,
        \"tuning_holdout_minimum_rows\": 20,
        \"tuning_holdout_minimum_rows_per_class\": 2,
        \"absolute_development_minimum_rows\": 30,
        \"absolute_development_minimum_rows_per_class\": 2,
        \"absolute_holdout_minimum_rows\": 10,
        \"absolute_holdout_minimum_rows_per_class\": 2,
        \"fallback_hyperparameters\": \"selected_all_customer\",
        \"performance_policy\": \"same_as_all_customer\"
      }
    }

Validation rejects overlapping boundaries, invalid minimums, unknown required family names, and class minimums larger than totals. Resolved configuration is copied into the immutable run directory.

## 13. Artifacts

Add these global run artifacts:

    subgroups/
      subgroup_definitions.json
      subgroup_assignments.parquet
      subgroup_eligibility.json
      subgroup_eligibility.csv
      subgroup_metrics.json
      subgroup_metrics_long.parquet
      subgroup_metrics_long.csv
      subgroup_confusion_matrices.parquet
      plots/
    customer_models/
      customer_model_eligibility.json
      customer_model_eligibility.csv
      <normalized-customer>/
        training_result.json
        locked_holdout_predictions.parquet
        metrics.json
        confidence_intervals.json
        candidate/

`subgroup_assignments.parquet` is a long table containing project key, population, family, band, source value, assignment status, and schema checksum. It is never model input.

Customer releases use:

    model_artifacts/schedule-risk/
      current.json
      releases/<all-customer-model-version>/
      customer-releases/
        <normalized-customer>/
          current.json
          releases/<customer-model-version>/

Each customer bundle records customer identity, parent global version, training mode, eligibility, hyperparameter origin, class counts, and customer performance gates. Existing checksum and atomic staging/rename semantics remain mandatory.

## 14. Current-Data Preflight

Preflight source:

`model_artifacts/schedule-risk/runs/schedule-rf-development-20260719T100419Z-ff12da52`

It contains 3,469 matched labeled projects: 2,785 development and 684 locked holdout. The qualified predictor schema has 3,632 fields. Class counts below are `no delay / mild delay / significant delay`.

### 14.1 Required non-customer families

| Family | Band | Development class counts | Holdout class counts | Eligibility |
|---|---|---:|---:|---|
| Duration | `<60` | 30 / 9 / 511 | 7 / 3 / 105 | Pass |
| Duration | `60-364` | 1060 / 67 / 818 | 278 / 16 / 200 | Pass |
| Duration | `>=365` | 227 / 16 / 47 | 56 / 9 / 10 | Pass |
| Planned value | `<=0` | 1047 / 15 / 34 | 262 / 3 / 10 | Pass |
| Planned value | `(0,$1M)` | 164 / 43 / 574 | 48 / 14 / 114 | Pass |
| Planned value | `[$1M,$10M)` | 68 / 30 / 699 | 23 / 9 / 171 | Pass |
| Planned value | `>=10M` | 38 / 4 / 69 | 8 / 2 / 20 | Pass, exact mild-class minimum |
| Item count | `1-19` | 947 / 31 / 264 | 246 / 7 / 58 | Pass |
| Item count | `20-49` | 284 / 39 / 582 | 74 / 11 / 126 | Pass |
| Item count | `>=50` | 86 / 22 / 530 | 21 / 10 / 131 | Pass |
| Missingness | `0%` | 238 / 76 / 1335 | 68 / 25 / 305 | Pass |
| Missingness | `>0%` | 1079 / 16 / 41 | 273 / 3 / 10 | Pass |

All four required families currently pass. Planned value `>=10M` is fragile: mildly delayed support is exactly at both minimums. Normal sampling changes could make it ineligible in a later run, so every candidate must rerun the gate.

Current duration ranges from 1 to 73,049 days. The extreme maximum warrants a source-quality warning but does not invalidate the `>=365` band. `PROJECTPLANNEDVALUE` has 1,371 nonpositive records and reaches $545,090,300.

### 14.2 Customer dispositions

| Customer | Development class counts | Holdout class counts | Planned disposition |
|---|---:|---:|---|
| Adams | 19 / 3 / 36 | 8 / 3 / 8 | Absolute floor passes; fit with all-customer hyperparameters |
| CCD | 111 / 12 / 30 | 30 / 5 / 4 | Independently tune |
| CLV | 2 / 0 / 0 | 0 / 0 / 0 | No customer model; absolute floor fails |
| Lincoln | 1124 / 47 / 287 | 287 / 14 / 62 | Independently tune |
| UDOT | 61 / 30 / 1023 | 16 / 6 / 241 | Independently tune |
| Amtrak | No matched labels | No matched labels | No customer model; labels unavailable |

These are eligibility dispositions, not performance results. Every fitted model must still pass its performance gate.

## 15. Implementation Work Breakdown

### Phase 0: Architecture and contracts

- [ ] Update `schedule_risk_model_training_pipeline_architecture.md` with all five families, eligibility semantics, and customer model lifecycle.
- [ ] Update `schedule_risk_agent_architecture.md` with dual-model loading, customer pointers, and MCP behavior.
- [ ] Version subgroup, customer-model, report, and response contracts.
- [ ] Link this plan from the result-availability plan.

Acceptance: customer isolation is not described as a customer-specific model, and no component chooses between outputs.

### Phase 1: Configuration

- [x] Add strict `SubgroupEvaluationConfig` and `CustomerModelConfig` models.
- [x] Add agreed defaults to example and development configurations.
- [x] Add boundary and support validators.
- [x] Store resolved configuration with every run.
- [ ] Unit test malformed and overlapping policies.

Acceptance: invalid policy fails before data loading or fitting.

### Phase 2: Assignment and eligibility

- [ ] Add `training_pipeline/subgroups.py` with pure assignment functions.
- [x] Calculate missingness from `numeric_matrix(joined, accepted)` before imputation.
- [x] Join assignments one-to-one to OOF and holdout predictions.
- [x] Implement all common family checks and detailed failures.
- [x] Write artifacts atomically.
- [x] Skip an entire family after any failed criterion.

Acceptance: every project has exactly one assignment per eligible family; an induced sparse band makes only its family ineligible.

### Phase 3: Metrics and reporting

- [x] Reuse `classification_metrics` per eligible band.
- [x] Add normalized metric and confusion serializers.
- [ ] Add bootstrap intervals with iteration accounting.
- [x] Add gate-first tables and comparison charts.
- [x] Emit `null + unavailable_reason` for unsupported metrics.
- [x] Remove the report statement that subgroup work is unimplemented.

Acceptance: HTML values reconcile to JSON/Parquet and no configured band is silently absent.

### Phase 4: Global release integration

- [x] Pass subgroup eligibility into `evaluate_release_gates`.
- [ ] Add `required_subgroup_family_eligibility`.
- [x] Retain rejected candidate artifacts.
- [ ] Verify failed promotion leaves `current.json` unchanged.
- [ ] Keep subgroup performance diagnostic pending approved thresholds.

Acceptance: current data passes all four family gates; induced failure blocks only new promotion.

### Phase 5: Customer-specific training

- [x] Evaluate every feature-snapshot customer, including no-label customers.
- [x] Implement tuning, fallback, and unavailable dispositions.
- [x] Tune eligible customers on customer development data only.
- [x] Fit fallback customers with frozen all-customer hyperparameters.
- [x] Fit nothing below the absolute floor.
- [x] Evaluate once on the existing customer holdout.
- [x] Apply global-equivalent metrics, parity, and integrity gates.
- [x] Write complete customer artifacts.

Acceptance: current data produces the six dispositions in Section 14.2 before performance gating, with no cross-customer training rows.

### Phase 6: Customer release management

- [x] Add verified customer promotion and rollback.
- [ ] Use customer-scoped atomic pointers.
- [ ] Require a released parent all-customer model.
- [ ] Preserve customer incumbents independently.
- [ ] Add registry discovery and integrity checks.

Acceptance: one customer's release operation cannot alter global or other-customer pointers.

### Phase 7: Runtime and MCP

- [x] Load global release as mandatory.
- [x] Load customer releases independently.
- [x] Build validated matrices for both models.
- [x] Return both slots with status and lineage.
- [x] Preserve batch partial success.
- [ ] Update MCP schemas, examples, and errors.

Acceptance: every successful project has an all-customer result and an explicit customer-specific result or unavailability.

### Phase 8: Validation and documentation

- [ ] Run unit and integration suites.
- [ ] Run the full current-data pipeline.
- [ ] Verify all four required family gates pass.
- [ ] Verify customer dispositions.
- [ ] Review generated HTML completeness.
- [ ] Verify checksums and serialization parity.
- [ ] Record runtimes and artifact sizes.
- [ ] Update operator and release documentation.

Acceptance: artifacts, report values, and runtime responses trace to the same immutable snapshots, splits, schema, and versions.

## 16. Test Matrix

### Unit tests

- Duration boundaries 59, 60, 364, and 365.
- Value boundaries negative, 0, positive, 999999.99, 1000000, 9999999.99, and 10000000.
- Item boundaries 1, 19, 20, 49, and 50.
- Missingness for finite, null, nonnumeric, and both infinities.
- Eligibility thresholds at one below, exact, and one above.
- Entire-family skip after one failed band.
- Customer tuning, fallback, and unavailable paths.
- Predictor exclusion of `CUSTOMERNAME`.

### Integration tests

- Prediction joins are one-to-one and complete.
- OOF metrics contain no in-fold predictions.
- Locked holdout remains untouched by tuning.
- Fallback parameters exactly match the parent global selection.
- Customer tuning uses customer development rows only.
- Metrics and reports reconcile.
- Undefined metrics include reasons.
- Required-family failure blocks production promotion.
- Customer failure does not block global or other customers.
- Rejected candidates preserve incumbent pointers.

### Runtime tests

- Global and customer models available.
- Global available and customer unavailable.
- Global available and customer bundle corrupt.
- Global unavailable or corrupt.
- Mixed-customer batch with mixed availability.
- No implicit model recommendation.
- Model/schema version mismatch.

## 17. Release Decision Matrix

| Condition | Global candidate | Customer candidate | Pointer behavior |
|---|---|---|---|
| Required non-customer family ineligible | Reject | Do not promote children of rejected parent | Keep all incumbents |
| Global metric or integrity gate fails | Reject | Do not promote children of rejected parent | Keep all incumbents |
| Global passes; customer floor fails | Promote global if otherwise approved | Unavailable | Update global only |
| Global passes; customer performance fails | Promote global if otherwise approved | Reject that customer | Keep customer incumbent |
| Global and customer both pass | Promote independently | Promote after parent | Update applicable pointers atomically |
| Another customer fails | No effect | No effect | Isolated pointers |

Production promotion still requires approved numeric performance thresholds and production-eligible label lineage. This plan does not reinterpret current development warnings as passes.

## 18. Completion Criteria

Implementation is complete only when:

1. All five evaluation families have explicit dispositions in every completed run.
2. All four required non-customer families produce full metrics or block new global production promotion with exact failed criteria.
3. Current data passes all four required-family eligibility gates.
4. Every customer receives a documented tuning, fallback, or unavailable disposition.
5. Runtime exposes all-customer and independently available customer-specific results without choosing between them.
6. Global, customer, and incumbent release behavior is tested.
7. Reports contain no unexplained missing, omitted, or `NaN` metrics.
8. Configuration, manifests, checksums, predictions, metrics, and reports are mutually traceable.
9. Architecture and operator documents describe implemented behavior.

