# Schedule Risk Training Result Availability Plan

## 1. Purpose

This plan separates results that were actually produced from results that are
limited, unavailable, not run, or blocked. It defines the work needed to make
each currently missing result properly available without estimating, filling,
or silently substituting a different test.

Status meanings:

- **Completed**: the configured procedure ran and produced inspectable artifacts.
- **Limited**: the procedure ran, but part of the intended population could not
  be evaluated.
- **Unavailable**: the procedure was requested, but required data was absent or
  insufficient.
- **Not run**: the procedure was not invoked or has not yet been implemented.
- **Blocked**: a release-level result cannot be claimed until external approval
  or environment prerequisites are satisfied.

## 2. Validated Baseline

Final validation run:

- Run ID: `schedule-rf-development-20260719T100419Z-ff12da52`
- Model version: `schedule-rf-20260719T100524Z-ff12da52`
- Run status: `development_candidate`
- Configured search: 2 of 2 candidates completed
- Cross-validation: 3-fold out-of-fold results completed
- Locked hash holdout: 684 projects completed
- Customer isolation: limited, with 4 of 6 feature-snapshot customers evaluated
- Feature importance: 3,632 qualified feature importances exported
- Temporal evaluation: completed on 694 newer projects
- Report: 12 embedded charts, no external image dependencies, and no unresolved
  report placeholders
- Automated validation: 12 tests passed

The two-candidate search completed exactly as configured. This is accurate
execution evidence, but it is not evidence of exhaustive hyperparameter
optimization.

## 3. Results Already Available

The following evidence is complete for the current development run:

1. Immutable feature and label snapshot checksums and manifests.
2. Feature-to-label reconciliation, including 2,293 feature-only projects.
3. Development and locked-holdout class counts.
4. Beginning-only feature qualification and rejection reasons.
5. All configured random-forest candidate results and selected parameters.
6. Development in-sample, CV out-of-fold, and locked-holdout metrics.
7. Locked-holdout confusion matrices, per-class metrics, ROC, precision-recall,
   calibration, and four bootstrap confidence intervals.
8. Leave-one-customer-out metrics for Adams, CCD, Lincoln, and UDOT.
9. Model feature importance and feature-family attribution.
10. Serialization parity, artifact hashes, reference profiles, and release-gate
    dispositions.

## 4. Temporal Evaluation

**Current status: Completed**

### Resolution completed

The earlier feature snapshot, `schedule-features-20260718T140401Z-fa9eac7d`,
predated the publisher change that retains `PLANNEDSTARTDATE` as split-only
metadata. A corrected immutable snapshot,
`schedule-features-20260719T094750Z-c7548ddb`, was created from Snowflake and
physically verified to contain the field. The field remains excluded from the
random-forest predictor matrix.

Snapshot validation found:

- 5,762 projects total;
- 5,751 projects with valid planned start dates;
- 11 projects with missing planned start dates, all from UDOT; and
- all six configured customers represented.

The development configuration was updated to the new immutable snapshot and the
complete training pipeline was rerun with temporal testing enabled. Temporal
population counts, class support, normalized metrics, an HTML metrics table, a
confusion matrix, and deterministic tests are now included.

### Validated temporal result

- Boundary date: `2024-03-21 06:00:00`
- Temporal training population: 2,775 older projects
- Temporal test population: 694 newer projects
- Test class support: 301 no-delay, 41 mild-delay, and 352 significant-delay
- Missing dates excluded from the labeled temporal population: 0
- Metrics-long population: `temporal_holdout`
- Macro F1: 0.5744
- Balanced accuracy: 0.5725
- Significant-delay recall: 0.8977
- OVR macro ROC AUC: 0.8416
- Log loss: 0.6579
- Expected calibration error: 0.0757

### Remaining data-quality follow-up

The 11 missing UDOT dates occur outside the currently matched labeled
population, so they did not reduce the temporal test. The feature data also has
at least one historical planned start date of `1900-01-01`. These values should
be reviewed with the client and covered by an explicit date-validity policy, but
they no longer prevent temporal evaluation.

## 5. Customer-Isolation Coverage

**Current status: Limited**

### Root causes

- **Amtrak:** present in the 5,762-row feature snapshot but absent from the legacy
  CSV label snapshot. It has no matched retrospective labels and cannot be
  evaluated.
- **CLV:** only 2 matched projects, both in class 0. It fails the configured
  minimum of 20 rows and minimum support of 2 projects in every class.
- Adams, CCD, Lincoln, and UDOT meet the configured support rules and were
  evaluated.

### Remediation

1. Generate a new label snapshot directly from Snowflake using
   `schedule_risk_label_calculation.sql` and the implemented
   `snapshot-labels-snowflake` command.
2. Produce a per-customer reconciliation from source projects through valid
   labels, with exclusion counts by reason.
3. For Amtrak, determine whether zero labels are caused by source joins/status
   filters, genuinely absent completed payment history, or missing access.
4. For CLV, determine whether additional historical projects exist and whether
   current filters unnecessarily remove them.
5. Do not synthesize classes or weaken support thresholds to force a metric.
   Retain `Unavailable` until real support exists.
6. Rerun customer-isolation tests after label coverage improves.

### Acceptance criteria

- Every feature-snapshot customer is listed in the report, including customers
  with zero labels.
- Evaluated, labeled-but-insufficient, and no-label customers are distinct.
- A customer's metrics are shown only when its holdout meets the configured row
  and all-class support requirements.

## 6. Planned Subgroup Evaluations

**Current status: Not run; implementation is missing**

The architecture requires subgroup metrics for planned-duration band,
planned-value band, contract-item-count band, and feature-missingness band. Only
customer isolation is currently implemented.

### Remediation

1. Define versioned, data-independent band boundaries in configuration.
2. Assign subgroups without adding subgroup identity to the predictor matrix.
3. Compute overall and per-class metrics, support, and instability reasons for
   each band on CV out-of-fold and locked-holdout predictions.
4. Enforce `minimum_subgroup_support` from the release policy.
5. Write `subgroup_metrics.json`, normalized metrics-long rows, tables, and
   comparative charts.
6. Add release-gate logic for unacceptable subgroup collapse after thresholds
   are approved.

### Acceptance criteria

- Every planned subgroup family has a disposition.
- Sparse groups are marked unstable or unavailable, never omitted.
- Band definitions and support counts are stored with the run.

## 7. Additional Confidence Intervals

**Current status: Partially available**

Bootstrap intervals currently exist for macro F1, balanced accuracy,
significant-delay recall, and selection score. The report correctly shows
`Not calculated` for OVR macro ROC AUC, log loss, and expected calibration error.
The bootstrap implementation does not currently collect those metrics.

### Remediation

1. Extend the stratified bootstrap metric list to ROC AUC, log loss, calibration
   error, Brier score, and selected per-class metrics.
2. Preserve attempted/valid iteration counts and undefined reasons.
3. Increase the production default from the current development setting of 100
   to the architecture target of 1,000 iterations, subject to runtime review.
4. Add deterministic interval tests and report-schema assertions.

## 8. Candidate-versus-Incumbent Comparison

**Current status: Not run; training integration is missing**

A local released bundle exists, but the training configuration has no incumbent
bundle input. The current `compare` command scores one bundle on a later labeled
dataset; it does not score candidate and incumbent on the same locked projects
or perform a paired bootstrap.

### Remediation

1. Add optional `incumbent_bundle_path` and comparison-policy fields to the run
   configuration.
2. Verify incumbent checksums and feature-schema compatibility.
3. Score both models on the candidate run's identical locked-holdout projects.
4. Produce paired bootstrap deltas and confidence intervals.
5. Write `comparison_to_incumbent.json` and a report section.
6. Enforce `allowable_incumbent_regression` only after that threshold is
   approved.

## 9. External Drift and Later-Data Performance

**Current status: Not run; no later labeled snapshot was supplied**

Drift cannot be inferred from the training snapshot compared with itself. The
external `compare` command and fixed reference histograms exist, but a later
feature snapshot plus mature retrospective labels are required.

### Remediation

1. Preserve the current release and reference profiles unchanged.
2. After enough projects in a later feature snapshot have completed, create a
   matching immutable Snowflake label snapshot.
3. Run the existing `compare` command on that later population.
4. Extend comparison output with configured pass/warn/fail dispositions,
   prediction-distribution drift, and confidence intervals.
5. Schedule repeated comparisons outside the training process.

## 10. Production Release Validation

**Current status: Blocked**

Independent blockers are:

- labels come from a legacy CSV and are permanently marked `development_only`;
- client approval of beginning-available features and target semantics is
  pending;
- all numeric release thresholds are null;
- incumbent and subgroup regression gates are not available;
- Docker client 26.1.3 is installed, but the process cannot connect to the Docker
  daemon at `/var/run/docker.sock`, so Python 3.11 container parity and service
  smoke tests have not run.

### Remediation order

1. Obtain feature and target-definition approval.
2. Generate and validate a production-eligible Snowflake label snapshot.
3. Complete subgroup and incumbent evaluations.
4. Approve numeric release thresholds using development evidence and business
   risk tolerances.
5. Restore Docker daemon access, build the image, run the full test suite in
   Python 3.11, and execute serialization and MCP smoke tests.
6. Run production promotion only when every required gate passes.

## 11. Delivery Sequence

1. **Completed - Temporal completeness:** refreshed features and validated the temporal holdout.
2. **P0 - Customer data completeness:** run Snowflake label extraction with per-customer reconciliation and rerun customer isolation.
3. **P1 - Evaluation completeness:** implement subgroup metrics and expanded
   confidence intervals.
4. **P1 - Release comparison:** implement paired candidate/incumbent evaluation.
5. **P2 - Later-data evidence:** collect a mature later snapshot and run drift and
   external performance comparison.
6. **P0 for production - Governance and runtime:** approve policy values and
   restore Docker daemon access.

No unavailable result should be converted to zero, `NaN` without a reason, or a
proxy metric. The report must continue to show the status, cause, affected
population, and exact prerequisite for every result.
