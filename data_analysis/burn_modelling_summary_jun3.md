# Burn Modelling Summary - June 3

## Context

This analysis uses `ci_payment_details_2.csv`, where each row is a payment event for a contract `ITEMID`. The revised file fixes the earlier grain issue: for every item, `NUM_PAYGROUPS` equals both the raw row count and the distinct `ITEMID + WPPOSTINGDATE` count.

The analysis focuses on the `_months` fields:

- `CI_BURN_RATE_MONTHS`: monthly linear baseline for the item.
- `THIS_BURN`: actual burn for the payment row.
- `BURN_DELTA_MONTHS`: absolute row delta from the monthly baseline.
- `BURN_DELTA_MONTHS_PERCENT`: normalized delta, approximately `(THIS_BURN - CI_BURN_RATE_MONTHS) / CI_BURN_RATE_MONTHS * 100`.

The `_pg` fields were treated as secondary reference fields and not used as the main modeling basis.

## Dataset Shape

`ci_payment_details_2.csv` contains:

- 16,435 payment rows.
- 875 distinct contract items.
- 13 fields.
- No row-count deviation from `NUM_PAYGROUPS` in the revised extract.

Paygroup reconciliation:

| Metric | Count |
|---|---:|
| Sum `NUM_PAYGROUPS` | 16,435 |
| Raw rows | 16,435 |
| Distinct `ITEMID + WPPOSTINGDATE` dates | 16,435 |
| Items where `rows == NUM_PAYGROUPS` | 875 / 875 |
| Items where `distinct posting dates == NUM_PAYGROUPS` | 875 / 875 |

This means the revised query now has a clean payment-event grain.

## Important Modelling Caveat

`CI_BURN_RATE_MONTHS` appears to be derived from completed item burn divided by modelled months, using roughly:

```text
modeled_months = (DAYS_BETWEEN + 30) / 30
CI_BURN_RATE_MONTHS = item_total_burn / modeled_months
```

That makes `CI_BURN_RATE_MONTHS` very useful for retrospective normalization, allocation, anomaly detection, and understanding burn shape.

However, it can leak future information in a forward forecasting model if the final item total is not known at prediction time. For true forecasting, item scale must be estimated from fields available before completion, such as item quantity, unit price, standard item/category, project/contract metadata, schedule, and historical behavior.

## Core Finding

A good model of burn should not treat each payment event as one full month of burn.

The current monthly baseline is useful, but it is not a complete row-level prediction model. Each row is a payment event, not necessarily a month of work. Most payment events are much smaller than one full monthly burn rate, while a smaller number are large bursts.

In the revised data:

- Median `THIS_BURN` is about 65,216.
- Median `CI_BURN_RATE_MONTHS` is about 207,983.
- Median `BURN_DELTA_MONTHS_PERCENT` is about -77.60%.
- Only 1,574 of 16,435 rows are within +/-25% of the monthly-linear baseline.
- 986 rows are more than +100% above the monthly baseline.
- 126 rows are below -100%, matching the negative-burn row count.
- Negative burn rows should be understood as corrections or reversals, not ordinary low-burn observations.

This is a bursty payment-event process, not a smooth monthly accrual process.

## Model Comparison Interpretation

The notebook compared several simple model families on held-out `ITEMID`s. Splitting by item matters because item-level scale is repeated across rows for an item; a row-level random split would leak item behavior across train and test.

### Current Monthly Rate Per Row

```text
prediction = CI_BURN_RATE_MONTHS
```

This is the current linear monthly baseline applied to every row.

It is systematically too high for most individual payment events. It assumes every payment event is equivalent to a full month of burn. That is why most `BURN_DELTA_MONTHS_PERCENT` values are negative.

This model is useful as a baseline for normalization, but not as a realistic row-level prediction model.

### Global Median Actual/Monthly Ratio

```text
prediction = CI_BURN_RATE_MONTHS * median(THIS_BURN / CI_BURN_RATE_MONTHS)
```

This performs better for the typical row because it learns that a normal payment event is a fraction of monthly burn, not a full month.

The median actual/monthly ratio is roughly 0.22, meaning the typical payment row is about 22% of the monthly burn baseline.

This model is simple and robust, but it does not conserve item totals and ignores payment timing.

### Global Mean Actual/Monthly Ratio

```text
prediction = CI_BURN_RATE_MONTHS * mean(THIS_BURN / CI_BURN_RATE_MONTHS)
```

This is less biased in aggregate dollars than the median-ratio model, but it is worse for the typical row because large positive bursts pull the mean upward.

The mean is sensitive to the heavy positive tail.

### Sequence and Calendar Bucket Ratio Models

Examples:

```text
prediction = CI_BURN_RATE_MONTHS * median_ratio(sequence_bucket)
prediction = CI_BURN_RATE_MONTHS * median_ratio(calendar_progress_bucket)
prediction = CI_BURN_RATE_MONTHS * median_ratio(sequence_bucket, item_size_bucket)
```

These models add payment position and timing shape. They improve interpretation and can modestly improve row-level predictions, but they do not solve the core issue by themselves.

The payment process is too bursty for sequence alone to create a smooth deterministic curve.

### Interval-Weighted Linear Allocation

A natural structural model is:

```text
exposure_months = 1.0 for the first payment
exposure_months = days_since_prior_payment / 30 for later payments
prediction = CI_BURN_RATE_MONTHS * exposure_months
```

This model aligns with the construction of `CI_BURN_RATE_MONTHS`, where the first payment effectively receives 30 days of length and the overall item duration is `(DAYS_BETWEEN + 30) / 30` months.

This model is the best structural allocation model because it nearly reconciles item totals. It spreads the known item total across the observed payment timeline in a coherent way.

However, it is not the best row-level point predictor. Many payment events occur close together, and payment dates appear to be administrative/accounting events rather than clean work-accrual boundaries. Very short intervals can cause this model to underpredict middle payments.

## Recommended Model

A good model should be hierarchical, scale-aware, and probabilistic.

### 1. Item Scale Layer

Estimate or accept an item-level total burn and duration.

For retrospective analysis, `CI_BURN_RATE_MONTHS` can supply this scale.

For forward forecasting, estimate scale from pre-completion features, such as:

- Contract item quantity.
- Unit price.
- Standard item number or item family.
- Project type.
- Contract type.
- Contract duration.
- Project schedule.
- Historical behavior for similar items.

### 2. Exposure Layer

Convert payment timing into exposure.

Recommended baseline:

```text
if first payment:
    exposure_months = 1.0
else:
    exposure_months = days_since_prior_payment / 30

structural_prediction = item_monthly_scale * exposure_months
```

This creates a total-conserving allocation when item monthly scale is derived from total burn and modeled duration.

### 3. Event-Shape Layer

Add a multiplicative event-shape factor:

```text
row_prediction = structural_prediction * event_shape_factor
```

Useful event-shape features include:

- Payment sequence number.
- Payment sequence fraction.
- Calendar progress fraction.
- Number of paygroups.
- Item duration.
- Time since prior payment.
- First/middle/last payment indicators.
- Item category or standard item prefix, if available.

Because the data is heavy-tailed, these factors should be robust:

- Median ratios.
- Quantile regression.
- Winsorized means.
- Regularized models.
- Tree/boosting models with quantile objectives, if richer features are added.

### 4. Correction/Reversal Layer

Negative `THIS_BURN` rows are real behavior and should be explicitly handled.

Recommended approach:

```text
correction_flag = THIS_BURN < 0
```

Then either:

- Model correction probability separately with a classifier, or
- Exclude corrections from the positive-burn model and add a separate correction process, or
- Keep them in a quantile model but add explicit negative/correction flags.

They should not be treated as ordinary low positive burn.

### 5. Uncertainty Layer

A single point prediction is not enough for this data.

The distribution has:

- A large mass below the monthly baseline.
- A heavy positive burst tail.
- Negative correction rows.
- Large item-to-item volatility.

A good model should return prediction intervals or quantiles, such as P10/P50/P90 burn, rather than only an expected value.

## Practical Formulas

### Retrospective Allocation / Anomaly Detection

Use this when the item total or `CI_BURN_RATE_MONTHS` is known and item-total reconciliation matters.

```text
base_monthly = CI_BURN_RATE_MONTHS

if first payment:
    exposure_months = 1.0
else:
    exposure_months = days_since_prior_payment / 30

structural_pred = base_monthly * exposure_months
row_pred = structural_pred * sequence_residual_factor
residual = THIS_BURN - row_pred
```

This is the best structure for asking whether a payment event is unusually high or low relative to the item timeline.

### Typical Payment-Event Prediction

Use this when row-level median accuracy matters more than item-total conservation.

```text
row_pred = CI_BURN_RATE_MONTHS * median_ratio(sequence_bucket, item_size_bucket)
```

This is robust and performs better for the typical payment event because it learns that most rows are only a fraction of the full monthly baseline.

### Forward Forecasting

Do not use completed-item `CI_BURN_RATE_MONTHS` if the final item total is unknown.

Instead:

```text
estimated_item_total = scale_model(item_quantity, unit_price, item_category, project_features, schedule_features)
estimated_duration_months = duration_model(schedule_features, historical_patterns)
estimated_monthly_scale = estimated_item_total / estimated_duration_months
row_pred = estimated_monthly_scale * exposure_months * event_shape_factor
```

This avoids leaking completed-item information into a forecast.

## Bottom Line

The burn process is not well described by evenly spreading monthly burn across payment rows.

Payment rows are bursty accounting/payment events. The strongest defensible model is a two-level model:

1. Estimate item-level scale.
2. Allocate or predict payment-event burn as a noisy, heavy-tailed share of that scale using exposure, sequence, and timing features.

Use interval-weighted allocation when item-total reconciliation matters. Use robust median or quantile ratio models when typical row prediction matters. Treat negative burns as explicit correction/reversal events. For real forecasting, replace `CI_BURN_RATE_MONTHS` with a separately estimated item scale built only from information available at prediction time.
