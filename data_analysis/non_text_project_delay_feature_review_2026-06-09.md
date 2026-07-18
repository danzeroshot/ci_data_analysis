# Non-Text Feature Design Review for Project Delay Modeling

Date: 2026-06-09  
Source data: `custpaydetails_all_project_contract_item_feature_dump_2026-06-09-1155.csv`

## Purpose

This document reviews likely non-text feature candidates from the broad project / contract / contract-item / payment-detail dump. The goal is to identify features that may predict project delay, especially features that can be captured at the beginning of the project after project, contract, and item records have been entered but before any payment postings are entered.

The target used in prior work is `PercentDelayed`:

```text
ActualDurationDays / PlannedDurationDays * 100 - 100
```

where actual duration is based on valid payment postings. Any feature that directly uses payment postings, pay estimates, actual start/end payment dates, actual duration, payment span, or completed burn should be treated as retrospective/leakage for early prediction.

## Dataset Context

The CSV parses to 585,539 rows with 74 columns. The line count is higher because long quoted text fields contain embedded newlines.

Unique entity counts from the parsed file:

| Entity | Count |
|---|---:|
| Projects | 5,762 |
| Contracts | 6,923 |
| Contract items | 227,066 |
| Parsed rows | 585,539 |

Customer row mix:

| Customer | Rows |
|---|---:|
| Lincoln | 287,510 |
| UDOT | 221,653 |
| Adams | 31,456 |
| Amtrak | 17,230 |
| CCD | 16,106 |
| CLV | 11,584 |

Project status row mix:

| ProjectStatus | Rows |
|---|---:|
| Construction | 280,198 |
| Complete | 259,542 |
| blank | 17,472 |
| In Progress | 13,688 |
| Funded | 10,691 |
| Awarded | 1,281 |
| Need Additional Funding | 893 |
| Advertisement | 593 |
| Design | 583 |
| No Work | 398 |

Payment validity flag:

| IsValidPostedPaymentRow | Rows |
|---|---:|
| 1 | 483,488 |
| 0 | 102,051 |

## Availability Classes

| Class | Meaning | Use in early-risk model? |
|---|---|---:|
| `early_project` | Available after project record is created | Yes |
| `early_contract` | Available after contract record is created | Yes, if prediction is after contract setup |
| `early_item` | Available after contract items are entered | Yes, if prediction is after item setup |
| `questionable_early` | Probably available, but workflow timing/customer consistency must be checked | Maybe |
| `post_payment` | Requires work postings, pay estimates, actual posting dates, or paid quantities | No for early model; yes for monitoring model |
| `target_or_leakage` | Target itself or direct component of target | No |
| `identifier_only` | IDs/codes useful for joins, grouping, and splits, but not as numeric predictors | No direct model use |

## Field Groups and Candidate Features

### 1. Identity, Grouping, and Split Fields

These should generally not be model predictors directly, but they are essential for grouping and reproducibility.

| Source fields | Candidate feature | Grain | Availability | Recommendation |
|---|---|---|---|---|
| `CUSTOMERNAME` | Customer fixed effect / one-hot / stratification | Project | `early_project` | Include for diagnostics and stratified models. Be careful: it can dominate pooled results. |
| `PROJECTID`, `PROJECTNAME` | Project key | Project | `identifier_only` | Use only as identifier, not predictor. |
| `PROJECTCODE` | Parsed project code tokens or code family | Project | `questionable_early` | Consider later. Project codes may encode customer/program/funding, but are inconsistent and partly numeric. |
| `CONTRACTID` | Contract grouping key | Contract | `identifier_only` | Use for aggregation only. |
| `ITEMID` | Item grouping key | Item | `identifier_only` | Use for aggregation only. |
| `STANDARDITEMNO` | Standard item family / prefix | Item | `early_item` | Strong candidate after careful parsing. Use prefixes/classes, not raw code IDs. |

Recommended inclusion:

- `CustomerName` as either one-hot or stratification variable.
- `StandardItemNoPrefix3`, `StandardItemNoPrefix5`, and `StandardItemNoFamily` if enough support exists across customers.
- Do not use raw project/contract/item IDs as predictors.

### 2. Project Status and Workflow Stage

`PROJECTSTATUS` is now a status name. It is nonempty in about 97% of rows and contains values like `Construction`, `Complete`, `In Progress`, `Funded`, and `Awarded`.

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `ProjectStatusAtExtract` | Project | `questionable_early` | Use cautiously | Status at extract time may be after payments and therefore not early-safe. |
| `IsProjectInitiallyFundedOrAwarded` | Project | `early_project` if captured at prediction date | Consider only if time-stamped status history exists | Current dump has only current status, not status-at-start. |
| `IsCompleteAtExtract` | Project | `target_or_leakage` for historical model | Exclude from early model | Completion status is strongly post-outcome. |

Recommendation: keep `PROJECTSTATUS` for dataset profiling and filtering, but do not include current extract status in early prediction unless we can reconstruct status as of project/contract setup.

### 3. Planned Schedule Fields

These are among the best early-available feature families because planned dates exist before payment postings.

Source fields:

- `CONTRACTSTARTDATE`
- `CONTRACTCLOSUREDATE`
- `PLANNEDSTARTDATE`
- `PLANNEDENDDATE`
- `HASVALIDCONTRACTSCHEDULE`
- `PLANNEDDURATIONDAYS`

Important profile notes:

- `CONTRACTSTARTDATE` is nearly complete: 99.98% nonempty.
- `CONTRACTCLOSUREDATE` is about 75.2% nonempty.
- `PLANNEDENDDATE` is about 67.9% nonempty at project target level.
- `PLANNEDDURATIONDAYS` ranges from negative values to very large outliers, so cleaning/winsorization is needed.

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `PlannedDurationDays` | Project | `early_project` / `early_contract` | Include | Strong prior model signal. Clean nonpositive and extreme values. |
| `HasValidPlannedDuration` | Project | `early_project` | Include | Missing/invalid schedules may themselves be risk signals. |
| `HasClosureDate` | Project/contract | `early_contract` | Include | Missing planned end date may signal poor setup quality. |
| `ContractPlannedDurationDays` | Contract | `early_contract` | Include aggregate stats | Use min/median/max/std across contracts. |
| `ProjectContractStartSpreadDays` | Project | `early_contract` | Include | Max contract start minus min contract start. Measures staggered contract setup. |
| `ProjectContractEndSpreadDays` | Project | `early_contract` | Include | Max closure minus min closure. Measures planned end dispersion. |
| `NumContractsWithInvalidSchedule` | Project | `early_contract` | Include | Schedule setup-quality signal. |
| `ShareContractsWithInvalidSchedule` | Project | `early_contract` | Include | Normalized version. |
| `PlannedStartMonth`, `PlannedStartQuarter` | Project | `early_project` | Include | Seasonality may matter. |
| `PlannedEndMonth`, `PlannedEndQuarter` | Project | `early_project` | Include if end exists | Captures winter/seasonal closeout risk. |
| `PlannedDurationBucket` | Project | `early_project` | Include | Tree models can infer buckets, but explicit bins help diagnostics. |

Recommended project-level schedule aggregates:

- `PlannedDurationDays`
- `Log1pPlannedDurationDays`
- `HasValidPlannedDuration`
- `HasPlannedEndDate`
- `PlannedStartMonth`
- `PlannedStartQuarter`
- `PlannedEndMonth`
- `PlannedEndQuarter`
- `NumContractsWithValidSchedule`
- `NumContractsWithInvalidSchedule`
- `ShareContractsWithInvalidSchedule`
- `MedianContractPlannedDurationDays`
- `MaxContractPlannedDurationDays`
- `StddevContractPlannedDurationDays`
- `ContractStartSpreadDays`
- `ContractEndSpreadDays`

### 4. Project and Contract Complexity

These are early-available after project/contract/item setup and likely important because project scope complexity tends to affect delay.

Source fields:

- `RAWCONTRACTCOUNT`
- `RAWCONTRACTITEMCOUNT`
- `ITEMID`
- `CONTRACTID`
- `ITEMCONTAINERID`
- `ITEMMODULEID`
- `COMMITMENTPOID`

Current dump includes `RawContractCount` and `RawContractItemCount`, but these are repeated across rows. In a modeling dataset, compute one project-level row.

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `NumContracts` | Project | `early_contract` | Include | More contracts may imply coordination complexity. |
| `NumContractItems` | Project | `early_item` | Include | Prior models found item/payment complexity predictive. |
| `ItemsPerContract` | Project | `early_item` | Include | Normalized complexity. |
| `NumCommitments` | Project | `early_contract` | Include | Usually available if commitments are set up before payment. |
| `ContractsPerProject` | Project | `early_contract` | Include | Same as `NumContracts`, useful naming. |
| `NumItemContainers` | Project | `early_item` | Include | Container/path structure may capture work breakdown complexity. |
| `ItemsPerContainer` | Project | `early_item` | Include | Work breakdown density. |
| `HasLinkedBudgetItems` | Project | `early_item` | Include cautiously | Linked budget fields are sparse but may be customer/workflow-specific. |
| `ShareItemsLinkedToBudget` | Project | `early_item` | Include cautiously | Sparse: linked budget rows only ~0.8% of raw rows. |

Recommended complexity features:

- `NumContracts`
- `NumContractItems`
- `Log1pNumContractItems`
- `ItemsPerContract`
- `NumCommitments`
- `NumItemContainers`
- `ItemsPerContainer`
- `HasAnyLinkedBudgetItems`
- `ShareItemsLinkedToBudget`

### 5. Planned Cost and Item Value Structure

These should be central early predictors. Prior models showed dollar and unit-price fields were important, but some earlier fields were computed from realized burn. The early-safe version should use planned item values: `ItemUnitPrice * ItemContractQuantity`.

Source fields:

- `ITEMUNITPRICE`
- `ITEMCONTRACTQUANTITY`
- `ITEMPLANNEDVALUE`
- `BUDGETITEMUNITPRICE`
- `BUDGETITEMQUANTITY`
- `BUDGETITEMPLANNEDVALUE`

Profile notes:

- Item unit price and quantity are complete, but contain negative values and extreme outliers.
- `ITEMPLANNEDVALUE` has negative values and very large maxima, so robust transforms are required.
- Linked budget fields are sparse, about 0.8% nonempty.

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `ProjectPlannedValue` | Project | `early_item` | Include | Sum of item planned values. Clean/winsorize. |
| `AbsProjectPlannedValue` | Project | `early_item` | Include | Robust if negative adjustments exist. |
| `Log1pAbsProjectPlannedValue` | Project | `early_item` | Include | Handles skew. |
| `DollarsPerPlannedDay` | Project | `early_item` + planned dates | Include | Early-safe analog of high-signal prior feature. |
| `DollarsPerContract` | Project | `early_item` | Include | Scope density per contract. |
| `DollarsPerContractItem` | Project | `early_item` | Include | Scope density per item. |
| `MedianItemPlannedValue` | Project | `early_item` | Include | Less outlier-sensitive than mean. |
| `MeanItemPlannedValue` | Project | `early_item` | Include | Use with log transform or winsorization. |
| `StddevItemPlannedValue` | Project | `early_item` | Include | Scope heterogeneity. |
| `MaxItemPlannedValue` | Project | `early_item` | Include | Large dominant item risk. |
| `MaxItemShareOfPlannedValue` | Project | `early_item` | Include | Concentration feature. |
| `ItemPlannedValueHerfindahl` | Project | `early_item` | Include | Measures concentration across items. |
| `ShareNegativePlannedValueItems` | Project | `early_item` | Include | Adjustment/change-order-like setup quality signal. |
| `ShareZeroPlannedValueItems` | Project | `early_item` | Include | Setup quality / placeholder item signal. |
| `NumExtremeUnitPriceItems` | Project | `early_item` | Include after thresholding | Detects possible data quality or specialized work. |
| `UnitPriceMedian`, `UnitPriceP90`, `UnitPriceMax` | Project | `early_item` | Include | Prior model used unit price features strongly. |
| `QuantityMedian`, `QuantityP90`, `QuantityMax` | Project | `early_item` | Include | Quantity scale and outliers. |

Recommended planned-cost features:

- `ProjectPlannedValue`
- `Log1pAbsProjectPlannedValue`
- `AbsProjectPlannedValue`
- `DollarsPerPlannedDay`
- `DollarsPerPlannedMonth`
- `DollarsPerContract`
- `DollarsPerContractItem`
- `MedianItemPlannedValue`
- `MeanItemPlannedValue`
- `StddevItemPlannedValue`
- `MaxItemPlannedValue`
- `MaxItemShareOfPlannedValue`
- `ItemPlannedValueHerfindahl`
- `ShareNegativePlannedValueItems`
- `ShareZeroPlannedValueItems`
- `UnitPriceMedian`, `UnitPriceP90`, `UnitPriceMax`, `UnitPriceStddev`
- `QuantityMedian`, `QuantityP90`, `QuantityMax`, `QuantityStddev`

### 6. Budget Linkage Fields

Linked budget fields are sparse in this dump, but the presence or absence of linkage may be informative about customer workflow or project setup quality.

Source fields:

- `LINKEDBUDGETITEMID`
- `BUDGETITEMID`
- `BUDGETSTANDARDITEMNO`
- `BUDGETMODULEID`
- `BUDGETCONTAINERID`
- `BUDGETITEMDESCRIPTION`
- `BUDGETITEMUNITPRICE`
- `BUDGETITEMQUANTITY`
- `BUDGETITEMPLANNEDVALUE`

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `HasAnyLinkedBudgetItems` | Project | `early_item` | Include cautiously | May mostly encode customer workflow. |
| `ShareItemsLinkedToBudget` | Project | `early_item` | Include cautiously | Sparse but normalized. |
| `NumBudgetModules` | Project | `early_item` | Consider | `BDGTEST` vs `BDGTREV` may indicate budget workflow. |
| `BudgetPlannedValueSum` | Project | `early_item` | Consider only where populated | Sparse; missingness must be explicit. |
| `BudgetToItemPlannedValueRatio` | Project | `early_item` | Consider only where populated | Risky due sparse coverage and denominator issues. |

Recommendation: include linkage presence/share fields as candidate predictors, but treat budget dollar features as optional and customer-controlled.

### 7. Standard Item Number Families

`STANDARDITEMNO` is complete and likely powerful. It should not be treated as a raw numeric field because values mix numeric and code forms such as `420-00133`, `625-00001`, and customer-specific formats.

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `StandardItemPrefix3` | Item/project aggregate | `early_item` | Include | Examples: `420`, `625`, `213`. |
| `StandardItemPrefix5` | Item/project aggregate | `early_item` | Include | More specific, but may be customer-specific. |
| `StandardItemDivision` | Project | `early_item` | Include if mappable | Requires mapping table or prefix grouping. |
| `NumDistinctStandardItemPrefixes` | Project | `early_item` | Include | Scope diversity. |
| `ShareTopPrefixItems` | Project | `early_item` | Include | Concentration of work type. |
| `TopNPrefixPresence` | Project | `early_item` | Include after frequency filtering | Similar to keyword features but structured. |

Recommendation: build a parallel feature family for standard item prefixes, similar to keyword coverage: count, share of items, and planned-value share by prefix.

### 8. Container / Work Breakdown Structure

`ITEMCONTAINERID` is present on 99.7% of rows. Container names/path are not in this dump, but ID structure still may indicate work breakdown complexity within a customer.

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `NumItemContainers` | Project | `early_item` | Include | Work breakdown complexity. |
| `ItemsPerContainer` | Project | `early_item` | Include | Density of breakdown. |
| `PlannedValuePerContainer` | Project | `early_item` | Include | Scope density by container. |
| `ContainerPlannedValueHerfindahl` | Project | `early_item` | Include | Concentration by WBS container. |

Recommendation: include aggregate container features, but not raw container IDs unless converted to per-customer frequent-category indicators.

### 9. Payment and Posting Fields

These are not suitable for early prediction, but they are useful for monitoring and model diagnostics after work begins.

Source fields:

- `WORKPOSTINGID`
- `WPPOSTINGDATE`
- `WORKPOSTINGSTATUS`
- `WORKPOSTINGQUANTITY`
- `WORKPOSTINGUNITRATE`
- `PAYESTIMATEID`
- `PAYESTIMATESTATUS`
- `PEDQUANTITY`
- `PEDAMOUNT`
- `CALCULATEDWORKCOMPLETEDAMOUNT`
- `HASWORKPOSTING`
- `HASPAYESTIMATEDETAIL`
- `HASPAYESTIMATE`
- `ISVALIDPOSTEDPAYMENTROW`
- `VALIDPOSTEDPROJECTWORKCOMPLETEDAMOUNT`
- `VALIDPOSTEDDETAILROWS`
- `VALIDPOSTEDPAYDATECOUNT`

| Candidate feature | Grain | Availability | Recommendation | Notes |
|---|---|---|---|---|
| `NumPayDates` | Project | `post_payment` | Exclude from early; include in monitoring | Very predictive but unavailable at start. |
| `PayDatesPerPlannedMonth` | Project | `post_payment` | Exclude from early; include in monitoring | Prior strongest signal. |
| `ActualStartLagDays` | Project | `target_or_leakage` / `post_payment` | Exclude early | Uses first payment date. |
| `ActualEndLagDays` | Project | `target_or_leakage` / `post_payment` | Exclude early | Uses final payment date. |
| `ValidPostedProjectWorkCompletedAmount` | Project | `post_payment` | Exclude early | Realized amount. |
| `ValidPostedDetailRows` | Project | `post_payment` | Exclude early | Payment activity count. |
| `WorkPostingStatus` / `PayEstimateStatus` | Row/project | `post_payment` | Exclude early | Workflow after payment begins. |
| `NegativePaymentShare` | Project | `post_payment` | Exclude early; include monitoring | Useful anomaly/change signal. |

Recommendation: maintain a separate “active monitoring” model that uses payment-derived fields, but keep these out of early project risk scoring.

### 10. Target and Direct Leakage Fields

These fields directly define or are derived from the target and should never be used as predictors in delay prediction.

| Field | Reason |
|---|---|
| `PERCENTDELAYED` | Target. |
| `ACTUALSTARTDATE` | Based on first payment date minus 30 days. |
| `ACTUALENDDATE` | Based on final payment date. |
| `FIRSTPOSTINGDATE` | Direct actual-start component. |
| `LASTPOSTINGDATE` | Direct actual-end component. |
| `ACTUALDURATIONDAYS` | Numerator of target. |
| `PAYMENTSPANDAYS` | Directly related to actual duration. |
| `VALIDPOSTEDPAYDATECOUNT` | Payment-derived; not target definition but post-outcome for completed projects. |

## Recommended Early-Available Feature Set

The recommended early feature set should be generated at project grain after project/contract/item setup, before payment postings. It should aggregate from contract/item rows to one row per `CustomerName + ProjectID`.

### Recommended Core Features

Identity/control:

- `CustomerName` one-hot or stratification flag
- Optional `ProjectStatusAtSetup`, only if status as of setup can be reconstructed

Schedule:

- `PlannedDurationDays`
- `Log1pPlannedDurationDays`
- `HasValidPlannedDuration`
- `HasPlannedEndDate`
- `PlannedStartMonth`
- `PlannedStartQuarter`
- `PlannedEndMonth`
- `PlannedEndQuarter`
- `NumContractsWithValidSchedule`
- `NumContractsWithInvalidSchedule`
- `ShareContractsWithInvalidSchedule`
- `MedianContractPlannedDurationDays`
- `MaxContractPlannedDurationDays`
- `StddevContractPlannedDurationDays`
- `ContractStartSpreadDays`
- `ContractEndSpreadDays`

Complexity:

- `NumContracts`
- `NumContractItems`
- `Log1pNumContractItems`
- `ItemsPerContract`
- `NumCommitments`
- `NumItemContainers`
- `ItemsPerContainer`
- `PlannedValuePerContainer`
- `ContainerPlannedValueHerfindahl`

Planned cost / item value:

- `ProjectPlannedValue`
- `AbsProjectPlannedValue`
- `Log1pAbsProjectPlannedValue`
- `DollarsPerPlannedDay`
- `DollarsPerPlannedMonth`
- `DollarsPerContract`
- `DollarsPerContractItem`
- `MedianItemPlannedValue`
- `MeanItemPlannedValue`
- `StddevItemPlannedValue`
- `MaxItemPlannedValue`
- `MaxItemShareOfPlannedValue`
- `ItemPlannedValueHerfindahl`
- `ShareNegativePlannedValueItems`
- `ShareZeroPlannedValueItems`
- `UnitPriceMedian`
- `UnitPriceP90`
- `UnitPriceMax`
- `UnitPriceStddev`
- `QuantityMedian`
- `QuantityP90`
- `QuantityMax`
- `QuantityStddev`

Budget linkage:

- `HasAnyLinkedBudgetItems`
- `ShareItemsLinkedToBudget`
- `NumBudgetModules`
- Optional `BudgetToItemPlannedValueRatio`, only with missingness flag

Standard item structure:

- `NumDistinctStandardItemPrefix3`
- `NumDistinctStandardItemPrefix5`
- `TopStandardItemPrefix3Count`
- `TopStandardItemPrefix3Share`
- `StandardItemPrefixHerfindahl`
- Top frequent prefix indicators or item-share fields after frequency/customer-coverage filtering

Keyword/text features:

- Project keyword presence/count features from reviewed keyword list
- Contract keyword presence/count features from reviewed keyword list
- Item keyword item-share and planned-value-share features from reviewed keyword list
- `customer_count` or cross-customer support should be used when selecting keyword features
- Place-name tokens should be flagged separately; include only as geography controls, not semantic work-scope features

### Recommended Exclusions for Early Model

Exclude from early project prediction:

- `WORKPOSTINGID`, `WPPOSTINGDATE`, `WORKPOSTINGSTATUS`
- `WORKPOSTINGQUANTITY`, `WORKPOSTINGUNITRATE`
- `PAYESTIMATEID`, `PAYESTIMATESTATUS`
- `PEDQUANTITY`, `PEDAMOUNT`, `CALCULATEDWORKCOMPLETEDAMOUNT`
- `HASWORKPOSTING`, `HASPAYESTIMATEDETAIL`, `HASPAYESTIMATE`, `ISVALIDPOSTEDPAYMENTROW`
- `VALIDPOSTEDPROJECTWORKCOMPLETEDAMOUNT`, `VALIDPOSTEDDETAILROWS`, `VALIDPOSTEDPAYDATECOUNT`
- `ACTUALSTARTDATE`, `ACTUALENDDATE`, `FIRSTPOSTINGDATE`, `LASTPOSTINGDATE`, `ACTUALDURATIONDAYS`, `PAYMENTSPANDAYS`
- Current/extract-time `PROJECTSTATUS` if it cannot be reconstructed as of project setup

## Recommended Next SQL/Pipeline Step

Create a project-level feature generator that produces one row per `CustomerName + ProjectID` with:

1. Early-safe schedule aggregates.
2. Early-safe item/contract complexity aggregates.
3. Early-safe planned cost and concentration features.
4. Standard item prefix coverage features.
5. Reviewed keyword features using customer coverage and place-name controls.
6. Explicit missingness flags for planned end dates, budget linkage, and invalid schedule rows.
7. No payment-posting or pay-estimate fields except in a separate monitoring dataset.

## Cautions

- The pooled all-customer dataset is imbalanced: Lincoln and UDOT dominate row counts. Model validation should include per-customer and leave-one-customer-out checks.
- Several planned cost and quantity fields have negative and extreme values. Use log transforms, winsorization, and explicit negative/zero flags.
- Current project status is likely contaminated by project lifecycle progress. Use only if status-at-prediction-time can be reconstructed.
- Some features are customer workflow proxies rather than universal project risk signals, especially linked budget fields and project code patterns.
