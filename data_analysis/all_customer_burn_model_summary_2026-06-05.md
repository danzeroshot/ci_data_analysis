# All-Customer Cumulative Burn Model Summary - June 5, 2026

## Executive Summary

The all-customer clustered cumulative curve dataset is suitable for continuing the Beta CDF burn-position modeling work, but the evidence is not uniform across customers. The model is strongest where there is enough history and where the customer's spend curve is materially non-linear relative to elapsed time. That is clearly true for the pooled all-customer dataset, UDOT, CCD, and Adams. Lincoln is the important exception: in the current held-out split, a pure linear cumulative spend model performs better than the Beta CDF variants.

The updated SQL output produced eligible modeled data for four customers: Adams, CCD, Lincoln, and UDOT. Amtrak and CLV are not present in the resulting modeled CSV after the current completion and eligibility filters. This does not necessarily mean those customers have no payment data; it means they did not survive this query's filters and thresholds in the delivered output.

The best general-purpose model remains the duration-bucket Beta CDF. On the pooled all-customer dataset, it improves MAE from 0.1640 under the linear model to 0.1454, a relative MAE improvement of about 11.4%. It also improves RMSE from 0.2294 to 0.1970, a relative RMSE improvement of about 14.1%. UDOT shows a larger improvement, CCD shows a meaningful improvement, Adams shows a small improvement, and Lincoln shows degradation.

## Prepared Data

| Scope | Rows | Items | Train Rows | Test Rows | Train Items | Test Items | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| All customers | 9,303 | 1,003 | 7,542 | 1,761 | 814 | 189 | Pooled Adams, CCD, Lincoln, UDOT |
| Adams | 606 | 45 | 414 | 192 | 35 | 10 | Small but usable; some correction behavior |
| CCD | 316 | 29 | 205 | 111 | 20 | 9 | Smallest modeled customer; clean cumulative curves |
| Lincoln | 750 | 54 | 625 | 125 | 43 | 11 | Linear model currently best |
| UDOT | 7,631 | 875 | 6,298 | 1,333 | 716 | 159 | Dominates the pooled dataset |

Data quality observations:

- The pooled dataset is heavily weighted toward UDOT: 7,631 of 9,303 rows, or about 82.0% of the observations.
- Adams, CCD, and Lincoln are much smaller. Their per-customer model results should be treated as early evidence, not final production-grade estimates.
- Negative/correction clusters are present in Adams, Lincoln, and UDOT. They create non-monotonic cumulative burn percentages in some item curves.
- CCD has no negative cluster rows in this output and no decreasing cumulative-percent item curves.
- Cumulative burn percentages can fall below 0 or exceed 1 because corrections and negative postings can make intermediate cumulative spend differ from final normalized item total.

## Distribution Characterization

The cumulative spend distribution is not a simple time-free marginal distribution. It is a bounded conditional distribution: `CumulativeBurnPct` must be interpreted relative to `ElapsedPct`. Across all scopes, elapsed percent and cumulative spend percent are strongly correlated, but the conditional spread is wide enough that a model needs tolerance bands or residual thresholds rather than a single deterministic curve.

| Scope | Mean Cumulative Spend | Median Cumulative Spend | Std Dev | Completed Edge Share | Pearson Corr | Spearman Corr | Distribution Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| All customers | 61.43% | 65.57% | 32.15% | 12.73% | 0.7664 | 0.7878 | Strong global structure with substantial scatter |
| Adams | 55.17% | 56.23% | 31.96% | 9.74% | 0.8583 | 0.8603 | Strong elapsed/spend relationship; less front-loaded than UDOT |
| CCD | 61.76% | 70.19% | 33.11% | 10.13% | 0.8621 | 0.8866 | Very strong monotone relationship despite small sample |
| Lincoln | 58.80% | 61.87% | 32.05% | 10.67% | 0.8049 | 0.8125 | Strong relationship, but current shape is closer to linear |
| UDOT | 62.18% | 66.69% | 32.07% | 13.27% | 0.7553 | 0.7817 | Largest and most heterogeneous customer; clear non-linear benefit |

The completed-edge share is expected because each completed contract item contributes a final cluster at or near 100% cumulative spend. That edge mass is not noise; it is structural. The main modeling implication is that the curve model should be judged mostly by conditional residuals and held-out item performance, not by the marginal histogram alone.

## Model Quality Summary

Errors are cumulative spend percentage-point errors. For example, MAE 0.145 means the average absolute position error is about 14.5 percentage points of final item spend.

| Scope | Best Model | Best MAE | Linear MAE | MAE Change vs Linear | Best RMSE | Linear RMSE | Suitability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| All customers | Duration-bucket Beta CDF | 0.1454 | 0.1640 | 11.4% better | 0.1970 | 0.2294 | Good pooled baseline, but UDOT-weighted |
| Adams | Duration-bucket Beta CDF | 0.1025 | 0.1064 | 3.6% better | 0.1460 | 0.1484 | Usable, but improvement is modest |
| CCD | Duration-bucket Beta CDF | 0.1089 | 0.1417 | 23.1% better | 0.1481 | 0.1817 | Promising, despite small sample |
| Lincoln | Linear cumulative spend | 0.1452 | 0.1452 | Beta worse | 0.1887 | 0.1887 | Use caution; customer appears closer to linear |
| UDOT | Duration-bucket Beta CDF | 0.1394 | 0.1760 | 20.8% better | 0.1940 | 0.2454 | Strongest evidence for Beta approach |

### All Customers

The pooled model performs well enough to be useful as a global default. The duration-bucket Beta CDF is the best model by MAE, RMSE, median absolute error, and P90 absolute error. It also substantially reduces the strong negative bias of the linear model. Linear has bias -0.0821, meaning it tends to overstate expected cumulative spend relative to actual position in this pooled sample; the duration-bucket Beta model has bias +0.0212.

However, the pooled model should not be interpreted as an equally representative all-customer model. UDOT dominates the row count and item count. The pooled curve is therefore closer to a UDOT-plus-adjustments model than a truly balanced customer model. It is still useful as a fallback or global benchmark, but customer-specific models should be preferred where enough customer data exists.

### Adams

Adams has 606 rows and 45 modeled items, with 10 held-out test items. The duration-bucket Beta CDF is best by MAE and RMSE, but the improvement over linear is small: MAE improves from 0.1064 to 0.1025. Linear has a slightly better P90 absolute error than the duration-bucket Beta model, so the Beta improvement is mostly average-case rather than tail-case.

This suggests Adams is suitable for a per-customer model, but it should be deployed conservatively. The current sample is small, and the Beta-vs-linear advantage is not large enough to justify ignoring the linear benchmark. A customer-specific Adams model should probably show both Beta expected position and linear expected position until more data confirms the shape.

### CCD

CCD has only 316 rows and 29 modeled items, but the results are surprisingly coherent. The duration-bucket Beta CDF improves MAE from 0.1417 under linear to 0.1089, a 23.1% improvement. RMSE improves from 0.1817 to 0.1481. The elapsed/cumulative correlation is also very strong, with Spearman correlation 0.8866.

The main limitation is sample size. CCD has only 20 training items and 9 test items, so held-out performance may be split-sensitive. Still, the shape is clean in this extract: no negative cluster rows and no decreasing cumulative burn curves. CCD is a good candidate for per-customer modeling, but confidence intervals or repeated item-level cross-validation are needed before production threshold tuning.

### Lincoln

Lincoln is the caution case. It has 750 rows and 54 modeled items, but the linear model is best by MAE and RMSE. The duration-bucket Beta CDF has MAE 0.1587 compared with linear MAE 0.1452, and it has a relatively large positive bias of 0.0772. That means the Beta curve is underestimating actual cumulative spend or expecting spend too slowly for the held-out Lincoln curves.

This does not mean the whole modeling approach fails for Lincoln. It means the current Beta parameterization, especially the duration-bucket version, does not match Lincoln's held-out spend shape as well as a simple linear assumption. Lincoln may have payment practices that are closer to regular progress burn, or the small training/test split may be producing unstable bucket parameters. For Lincoln, the current recommendation is to keep linear as the primary expected-position model until a Lincoln-specific curve family or more robust cross-validation proves otherwise.

### UDOT

UDOT remains the strongest evidence for the non-linear curve model. It has 7,631 rows and 875 modeled items, which gives the most reliable customer-specific estimate. The duration-bucket Beta CDF improves MAE from 0.1760 under linear to 0.1394, a 20.8% improvement. RMSE improves from 0.2454 to 0.1940. Linear also has a strong negative bias of -0.1011, while duration-bucket Beta bias is only +0.0054.

This confirms the original UDOT finding: elapsed time alone is too rigid for this customer. A duration-aware Beta CDF expected-position curve is materially better for estimating where an item sits relative to expected cumulative spend.

## Proxy Label / ROC Results

The proxy labels are generated by a separate anchored polynomial process, then consumed by the Beta-vs-linear model notebooks. These labels are not true budget-overrun or time-overrun outcomes; they are retrospective spend-position proxies. They are useful for comparing score behavior, but MAE/RMSE/bias remain the primary model-quality measures.

| Scope | Fast Proxy AUC: Beta | Fast Proxy AUC: Linear | Slow Proxy AUC: Beta | Slow Proxy AUC: Linear | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| All customers | 0.9901 | 0.9870 | 0.9825 | 0.9756 | Beta slightly better on both proxy tasks |
| Adams | 0.9957 | 1.0000 | 0.9546 | 1.0000 | Linear proxy AUC is perfect, but sample is small |
| CCD | 0.9994 | 1.0000 | 0.9987 | 0.9875 | Both excellent; Beta better on slow proxy |
| Lincoln | 1.0000 | 0.9994 | 0.9854 | 0.9812 | Both excellent; MAE still favors linear |
| UDOT | 0.9860 | 0.9835 | 0.9797 | 0.9647 | Beta better, especially on slow-spend proxy |

The ROC results are directionally supportive but should not override the error metrics. For example, Lincoln has excellent Beta proxy AUC but worse MAE/RMSE than linear. That means the Beta residual ranking may separate proxy positives well, while the expected-position level is still biased. For operational burn-position modeling, calibration matters at least as much as ranking.

## Suitability Assessment

### Suitable Now

UDOT is suitable for a customer-specific duration-bucket Beta CDF model. The evidence is strong: large sample, clear improvement over linear, substantially reduced bias, and consistent proxy-ranking performance.

The pooled all-customer model is suitable as a global benchmark and fallback model. It improves over linear and can support broader exploratory reporting. It should not be the only production model because it is heavily UDOT-weighted.

CCD is suitable for continued per-customer modeling, but with caution due to small sample size. The improvement over linear is large enough to justify more validation.

### Suitable With Caution

Adams is suitable for a per-customer model, but the improvement over linear is modest. It should be treated as a customer-specific model candidate, not yet a definitive replacement for linear.

### Not Yet Suitable as Beta Primary

Lincoln should not use the current duration-bucket Beta CDF as the primary expected-position model. Linear is currently better on held-out MAE/RMSE and has much lower bias. Lincoln needs either a linear default, a Lincoln-specific curve, or additional validation to determine whether the current split is misleading.

## Limitations

1. Customer coverage is incomplete.

   Amtrak and CLV are absent from the output after the current SQL filters. Before claiming all-customer coverage, we need to determine whether they lack qualifying completed items, fail the `CICostSize > 500000` and `COUNT(*) > 6` thresholds, have schema/query differences, or were filtered out by status/date logic.

2. The pooled model is UDOT-dominated.

   UDOT contributes about 82% of rows and 87% of modeled items. A pooled model trained directly on all rows is therefore not balanced across customers. If the goal is a customer-neutral global model, training should use customer weighting or item-level weighting.

3. Small-customer test sets are small.

   Adams, CCD, and Lincoln have only 10, 9, and 11 test items respectively. Their per-customer model rankings may be sensitive to the train/test split. Repeated item-level cross-validation is needed.

4. Negative and correction postings complicate monotonic cumulative curves.

   UDOT, Adams, and Lincoln have negative cluster rows and decreasing cumulative burn percentages for some items. A Beta CDF is monotonic by construction, while actual cumulative spend can temporarily reverse. The model should be understood as an expected-position smoother, not a literal reconstruction of every correction event.

5. The current target is final-normalized historical spend.

   The model uses completed historical item totals to normalize cumulative burn. For live forecasting, the denominator will need to be authorized budget, current committed budget, estimated final quantity, or another operational denominator. The production interpretation changes depending on that denominator.

6. Proxy labels are not true outcomes.

   Fast and slow proxy labels are based on deviation from a polynomial reference curve. They are helpful for model comparison, but they are not actual budget overrun, schedule overrun, claim, delay, or scope-change outcomes.

7. Duration and cluster-count buckets may be unstable for small customers.

   The duration-bucket Beta CDF works well for pooled data and UDOT, but for small customers the bucket-level fits can be noisy. Lincoln may be an example of this problem.

## Recommended Next Steps

1. Diagnose missing Amtrak and CLV rows.

   Run count diagnostics at each SQL stage by customer: raw payment rows, rows with non-null posting/pay-estimate status, completed contracts, item/date aggregated rows, eligible items above the burn and paygroup thresholds, and final clustered rows. This will show whether the absence is data reality or query/filter behavior.

2. Add customer-level model selection.

   Do not force the same model onto every customer. Use a simple rule: if a customer-specific Beta model beats linear by a material threshold under cross-validation, use Beta; otherwise use linear or pooled fallback. Lincoln should currently select linear.

3. Run repeated item-level cross-validation.

   The current train/test split is deterministic and useful, but small customers need repeated folds. Evaluate MAE, RMSE, P90AE, and bias across multiple item-level splits.

4. Evaluate weighted global models.

   Build at least three pooled versions: row-weighted, item-weighted, and customer-balanced. The current pooled model is row-weighted and UDOT-heavy.

5. Add calibration diagnostics.

   For each customer, plot residual by elapsed bucket, duration bucket, cluster count, item total size, and calendar period. The goal is to identify where the model is systematically too fast or too slow.

6. Treat corrections explicitly.

   Add flags such as `HasNegativeCluster`, `HasCumulativeDecrease`, and correction magnitude. Evaluate model quality separately for clean monotonic items versus correction-heavy items.

7. Connect to operational denominators.

   Decide how live expected position will be computed: against final estimated cost, current budget, original budget, revised budget, or contract quantity. This is necessary before using the model for budget-overrun alerting.

8. Develop threshold policy per customer.

   Position thresholds should not be universal initially. A 15 percentage-point residual may mean different things for UDOT versus Adams or Lincoln. Use customer-specific residual distributions to set warning and critical thresholds.

9. Keep linear as a visible benchmark.

   Even when Beta is primary, linear should remain visible in reports. It is interpretable and exposes customers like Lincoln where the non-linear curve may not add value.

## Bottom Line

The model is suitable as a serious expected-spend-position approach, especially for UDOT and the pooled dataset. It is not yet a one-size-fits-all customer model. The strongest practical architecture is a model-selection framework: customer-specific Beta where validated, linear where it wins, and a pooled/customer-balanced fallback where customer data is insufficient.

For the current output:

- Use duration-bucket Beta CDF as primary for UDOT.
- Use duration-bucket Beta CDF as the pooled all-customer benchmark, with the caveat that it is UDOT-weighted.
- Continue validating CCD because the model improvement is strong but the sample is small.
- Use Adams cautiously; improvement exists but is modest.
- Use linear as the primary Lincoln baseline until further validation supports a non-linear Lincoln model.
