# Project Feature Non-Keyword Field Dictionary

Date: 2026-06-10

Source dataset: `custpaydetails_project_feature_table_with_keywords_materialized_2026-06-09-1526_flat.csv`

This review excludes keyword-derived columns (`PROJ_KW_*`, `CONTRACT_KW_*`, `ITEM_KW_*`). It includes the remaining project-level identity, schedule, money, quantity, complexity, budget-linkage, and target fields so the client can verify field meanings and beginning-of-project availability.

Beginning-of-project assumption: after project, contract, and contract item setup is complete, but before work postings, pay estimates, or payments exist.

## Summary

- Non-keyword fields reviewed: 75
- Rows profiled: 5,762
- Keyword fields excluded: 6,000

### Availability Counts

| BeginningAvailable | Field Count |
|---|---:|
| Yes | 43 |
| No | 14 |
| Maybe | 6 |
| Yes, if planned duration valid | 2 |
| Yes, if planned end date exists | 2 |
| Yes, text field | 2 |
| Maybe / generally not early-safe as currently extracted | 1 |
| Yes, but excluded for customer-agnostic model | 1 |
| Yes, but identifier/code-like | 1 |
| Yes, identifier only | 1 |
| Yes, if closure date is entered at setup | 1 |
| Yes, if closure dates entered | 1 |

### Field Group Counts

| FieldGroup | Field Count |
|---|---:|
| planned schedule | 18 |
| target/post-payment | 13 |
| scope complexity | 9 |
| budget linkage | 5 |
| identity/project | 4 |
| item quantity | 4 |
| item unit price | 4 |
| planned money/item distribution | 4 |
| planned money | 3 |
| identity/export | 2 |
| planned money/data quality | 2 |
| planned money/item concentration | 2 |
| planned money/schedule | 2 |
| planned money/scope | 2 |
| workflow/status | 1 |

## Recommended Early-Available Non-Text Feature Families

- Planned schedule: planned start/end, planned duration, schedule validity, contract duration dispersion, planned start/end seasonality.
- Scope complexity: number of contracts/items/containers, items per contract/container, standard item prefix diversity.
- Planned value and item distribution: project planned value, absolute planned value, dollars per planned day/month/contract/item, median/mean/stddev/max item planned value, concentration metrics, zero/negative planned value shares.
- Unit price and quantity distributions: median, p90, max, and standard deviation. These need client validation because units vary across items.
- Budget linkage: potentially useful but marked `Maybe` because workflow timing and sparsity must be validated.

## Fields That Should Not Be Used For Beginning Prediction

- `PERCENTDELAYED` is the outcome label.
- `TARGET*` fields depend on valid posted payment rows or are computed only over payment-bearing rows.
- `PROJECTSTATUS` is current status at extract time; it is only early-safe if status history can reconstruct status as of the prediction date.
- Identifiers such as `PROJECTID` and `RECORD_ID` should be used for joining/auditing, not as predictors.
- `CUSTOMERNAME` is known at setup but was intentionally excluded from the latest customer-agnostic model.

## Client Validation Priorities

1. Confirm `CM.StartDt` and `CM.ClosureDt` are planned/original dates and whether they are overwritten during execution.
2. Confirm contract item `UnitPrice` and `ContractQuantity` represent setup-time planned values and whether revisions overwrite them.
3. Confirm whether commitments and budget links are created before any payment postings.
4. Confirm the business meaning of zero and negative planned-value items.
5. Confirm whether current `ProjectStatus` is extract-time only, and whether status-at-start can be reconstructed.
6. Confirm target definition: actual start as first posting date minus 30 days, actual end as last posting date.

## Identity/Export

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `RECORD_ID` | identifier | No | Sequential row number added in the Snowflake JSON export query, not a business field. | ROW_NUMBER() OVER (ORDER BY 1) in export wrapper. | None; exclude from modeling. |
| `CUSTOMERNAME` | categorical | Yes, but excluded for customer-agnostic model | Customer/source environment name: Amtrak, UDOT, CLV, CCD, Adams, or Lincoln. | Hardcoded per source schema branch in SQL. | Confirm whether future production use should remain customer-agnostic or allow customer-specific calibration. |

## Identity/Project

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `PROJECTCODE` | text/code | Yes, but identifier/code-like | Project code. | PM.ProjectCode aggregated as MAX at project grain. | Does project code encode schedule, funding, geography, or other future/outcome information? |
| `PROJECTDESCRIPTION` | text | Yes, text field | Project description. | PM.Description aggregated as MAX at project grain. | Confirm description is stable and available before payments. |
| `PROJECTID` | identifier | Yes, identifier only | Project primary key from source project table. | PM.ProjectID. | Use only for joins/audits, not modeling. |
| `PROJECTNAME` | text | Yes, text field | Project name. | PM.ProjectName aggregated as MAX at project grain. | Confirm project name is entered before contract item setup. |

## Workflow/Status

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `PROJECTSTATUS` | categorical | Maybe / generally not early-safe as currently extracted | Current project status name from status lookup. | LPS.StatusName joined from PM.StatusId and aggregated as MAX. | Can status history be reconstructed as of prediction date? If not, exclude from early model. |

## Planned Schedule

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `CONTRACTSTARTSPREADDAYS` | duration_days | Yes | Spread in planned contract start dates within project. | DATEDIFF(day, MIN(ContractStartDate), MAX(ContractStartDate)). | Confirm multiple-contract projects are expected and contract start dates are planned dates. |
| `HASPLANNEDENDDATE` | flag | Yes | Flag indicating the project has at least one planned end/closure date. | IFF(MAX(ContractClosureDate) IS NOT NULL, 1, 0). | Confirm missing closure date indicates incomplete setup vs valid open-ended project. |
| `HASVALIDPLANNEDDURATION` | flag | Yes | Flag indicating planned start and end exist and planned duration is positive. | IFF(MIN(ContractStartDate) IS NOT NULL AND MAX(ContractClosureDate) IS NOT NULL AND DATEDIFF(day, MIN(start), MAX(end)) > 0, 1, 0). | Confirm nonpositive planned durations should be treated as invalid setup. |
| `NUMCONTRACTSWITHINVALIDSCHEDULE` | count | Yes | Number of contracts with missing or nonpositive planned schedule. | COUNT(DISTINCT ContractID where start is null OR end is null OR DATEDIFF(day,start,end)<=0). | Confirm invalid schedule interpretation. |
| `NUMCONTRACTSWITHVALIDSCHEDULE` | count | Yes | Number of contracts with non-null start/end dates and positive duration. | COUNT(DISTINCT ContractID where start/end not null and DATEDIFF(day,start,end)>0). | Confirm invalid schedules reflect data quality/setup risk. |
| `PLANNEDSTARTDATE` | date | Yes | Earliest planned contract start date across contracts in the project. | MIN(ContractStartDate) across distinct contracts for the project. | Confirm CM.StartDt is planned/original start and not overwritten later. |
| `PLANNEDSTARTMONTH` | calendar | Yes | Calendar month of planned project start. | EXTRACT(month FROM MIN(ContractStartDate)). | Confirm month seasonality is acceptable for modeling. |
| `PLANNEDSTARTQUARTER` | calendar | Yes | Calendar quarter of planned project start. | EXTRACT(quarter FROM MIN(ContractStartDate)). | Confirm quarter seasonality is acceptable for modeling. |
| `SHARECONTRACTSWITHINVALIDSCHEDULE` | share | Yes | Fraction of contracts with missing or invalid planned schedule. | NumContractsWithInvalidSchedule / NULLIF(NumContracts, 0). | Confirm invalid schedule interpretation. |
| `CONTRACTENDSPREADDAYS` | duration_days | Yes, if closure dates entered | Spread in planned contract end/closure dates within project. | DATEDIFF(day, MIN(ContractClosureDate), MAX(ContractClosureDate)). | Confirm closure dates are planned dates and not actual closeout dates. |
| `LOG1PPLANNEDDURATIONDAYS` | numeric_transform | Yes | Log transform of nonnegative planned duration. | LN(1 + GREATEST(PlannedDurationDays, 0)). | Confirm whether negative planned durations should be nulled instead of clipped to zero for log transform. |
| `MAXCONTRACTPLANNEDDURATIONDAYS` | duration_days_stat | Yes | Maximum planned contract duration within project. | MAX(DATEDIFF(day, ContractStartDate, ContractClosureDate)) for contracts with non-null dates. | Confirm outlier contract durations are meaningful. |
| `MEDIANCONTRACTPLANNEDDURATIONDAYS` | duration_days_stat | Yes | Median planned duration across contracts in the project. | MEDIAN(DATEDIFF(day, ContractStartDate, ContractClosureDate)) for contracts with non-null dates. | Confirm contract-level planned duration definition. |
| `PLANNEDDURATIONDAYS` | duration_days | Yes | Planned project duration in days from earliest contract start to latest contract closure. | DATEDIFF(day, MIN(ContractStartDate), MAX(ContractClosureDate)). | Confirm desired planned duration definition: project-level contract envelope vs official PM project dates. |
| `PLANNEDENDDATE` | date | Yes, if closure date is entered at setup | Latest planned contract closure/end date across contracts in the project. | MAX(ContractClosureDate) across distinct contracts for the project. | Confirm CM.ClosureDt is planned end date and whether it changes during execution. |
| `PLANNEDENDMONTH` | calendar | Yes, if planned end date exists | Calendar month of planned project end. | EXTRACT(month FROM MAX(ContractClosureDate)). | Confirm planned end date is available before work starts. |
| `PLANNEDENDQUARTER` | calendar | Yes, if planned end date exists | Calendar quarter of planned project end. | EXTRACT(quarter FROM MAX(ContractClosureDate)). | Confirm planned end date is available before work starts. |
| `STDDEVCONTRACTPLANNEDDURATIONDAYS` | duration_days_stat | Yes | Standard deviation of planned contract durations within project. | STDDEV(DATEDIFF(day, ContractStartDate, ContractClosureDate)) for contracts with non-null dates. | Confirm dispersion across contracts is meaningful. |

## Scope Complexity

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `ITEMSPERCONTAINER` | ratio | Yes | Average number of contract items per item container. | COUNT(DISTINCT ItemID) / NULLIF(COUNT(DISTINCT ItemContainerID), 0). | None. |
| `ITEMSPERCONTRACT` | ratio | Yes | Average number of contract items per contract. | NumContractItems / NULLIF(NumContracts, 0). | None. |
| `LOG1PNUMCONTRACTITEMS` | numeric_transform | Yes | Log transform of contract item count. | LN(1 + COUNT(DISTINCT ItemID)). | None. |
| `NUMCOMMITMENTS` | count | Maybe | Number of distinct commitment purchase/order IDs linked to contracts. | COUNT(DISTINCT CommitmentPOID). | Are commitments always created before any work postings/payments? |
| `NUMCONTRACTITEMS` | count | Yes | Number of distinct contract items on the project. | COUNT(DISTINCT ItemID). | Confirm prediction point is after contract items are entered. |
| `NUMCONTRACTS` | count | Yes | Number of distinct contracts on the project. | COUNT(DISTINCT ContractID). | Confirm prediction point is after all initial contracts are entered. |
| `NUMDISTINCTSTANDARDITEMPREFIX3` | count | Yes | Number of distinct 3-character prefixes from standard item numbers. | COUNT(DISTINCT REGEXP_SUBSTR(StandardItemNo, ^[A-Za-z0-9]{3})). | Confirm prefix length is meaningful across customers. |
| `NUMDISTINCTSTANDARDITEMPREFIX5` | count | Yes | Number of distinct 5-character prefixes from standard item numbers. | COUNT(DISTINCT REGEXP_SUBSTR(StandardItemNo, ^[A-Za-z0-9]{5})). | Confirm prefix length is meaningful across customers. |
| `NUMITEMCONTAINERS` | count | Yes | Number of distinct item containers/work-breakdown containers used by project items. | COUNT(DISTINCT ItemContainerID). | Confirm container structure is stable at setup. |

## Planned Money

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `ABSPROJECTPLANNEDVALUE` | currency | Yes | Sum of absolute planned contract item values for the project. | SUM(ABS(COALESCE(ItemUnitPrice * ItemContractQuantity, 0))) across project items. | Confirm negative item values are adjustments and should contribute by absolute value. |
| `LOG1PABSPROJECTPLANNEDVALUE` | numeric_transform | Yes | Log transform of absolute project planned value. | LN(1 + ABS(AbsProjectPlannedValue)). | None. |
| `PROJECTPLANNEDVALUE` | currency | Yes | Sum of planned contract item values for the project; signed. | SUM(COALESCE(ItemUnitPrice * ItemContractQuantity, 0)) across project items. | Confirm item unit price and quantity represent original planned value at setup and are not overwritten later. |

## Planned Money/Schedule

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `DOLLARSPERPLANNEDDAY` | currency_rate | Yes, if planned duration valid | Planned project value divided by planned duration in days. | ProjectPlannedValue / NULLIF(PlannedDurationDays, 0). | Confirm signed planned value should be used vs absolute planned value. |
| `DOLLARSPERPLANNEDMONTH` | currency_rate | Yes, if planned duration valid | Planned project value divided by planned duration in approximate months. | ProjectPlannedValue / NULLIF(PlannedDurationDays / 30.0, 0). | Confirm 30-day month convention. |

## Planned Money/Scope

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `DOLLARSPERCONTRACT` | currency_rate | Yes | Planned project value per contract. | ProjectPlannedValue / NULLIF(NumContracts, 0). | Confirm signed planned value should be used vs absolute planned value. |
| `DOLLARSPERCONTRACTITEM` | currency_rate | Yes | Planned project value per contract item. | ProjectPlannedValue / NULLIF(NumContractItems, 0). | Confirm signed planned value should be used vs absolute planned value. |

## Planned Money/Item Distribution

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `MAXITEMPLANNEDVALUE` | currency_stat | Yes | Largest signed planned item value within the project. | MAX(ItemUnitPrice * ItemContractQuantity). | Confirm signed max is desired; negative-only projects may behave oddly. |
| `MEANITEMPLANNEDVALUE` | currency_stat | Yes | Mean planned value of contract items. | AVG(ItemUnitPrice * ItemContractQuantity). | Confirm current item values are original planned values. |
| `MEDIANITEMPLANNEDVALUE` | currency_stat | Yes | Median planned value of contract items. | MEDIAN(ItemUnitPrice * ItemContractQuantity). | Confirm current item values are original planned values. |
| `STDDEVITEMPLANNEDVALUE` | currency_stat | Yes | Standard deviation of planned item values. | STDDEV(ItemUnitPrice * ItemContractQuantity). | Confirm outliers are real scope values vs data issues. |

## Planned Money/Item Concentration

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `ITEMPLANNEDVALUEHERFINDAHL` | concentration_index | Yes | Herfindahl concentration index of absolute planned item values. | SUM(POWER(ABS(ItemPlannedValue) / NULLIF(AbsProjectPlannedValue, 0), 2)). | Confirm absolute-value concentration is appropriate. |
| `MAXITEMSHAREOFPLANNEDVALUE` | share | Yes | Largest absolute item value as a share of total absolute planned value. | MAX(ABS(ItemPlannedValue) / NULLIF(AbsProjectPlannedValue, 0)). | Confirm concentration is meaningful and absolute value is preferred. |

## Planned Money/Data Quality

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `SHARENEGATIVEPLANNEDVALUEITEMS` | share | Yes | Fraction of contract items with negative planned value. | COUNT(DISTINCT ItemID where ItemPlannedValue < 0) / NULLIF(NumContractItems, 0). | Ask client what negative planned-value items represent. |
| `SHAREZEROPLANNEDVALUEITEMS` | share | Yes | Fraction of contract items with zero planned value. | COUNT(DISTINCT ItemID where ItemPlannedValue = 0) / NULLIF(NumContractItems, 0). | Ask client what zero planned-value items represent. |

## Item Unit Price

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `UNITPRICEMAX` | currency_stat | Yes | Maximum item unit price across contract items. | MAX(ItemUnitPrice). | Confirm unit price outliers are meaningful. |
| `UNITPRICEMEDIAN` | currency_stat | Yes | Median item unit price across contract items. | MEDIAN(ItemUnitPrice). | Confirm unit price is original setup value, not revised current value. |
| `UNITPRICEP90` | currency_stat | Yes | 90th percentile item unit price across contract items. | PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY ItemUnitPrice). | Confirm unit price outliers are meaningful. |
| `UNITPRICESTDDEV` | currency_stat | Yes | Standard deviation of item unit prices. | STDDEV(ItemUnitPrice). | Confirm mixing different units makes this feature interpretable enough. |

## Item Quantity

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `QUANTITYMAX` | quantity_stat | Yes | Maximum contract item quantity. | MAX(ItemContractQuantity). | Confirm quantity outliers are meaningful. |
| `QUANTITYMEDIAN` | quantity_stat | Yes | Median contract item quantity. | MEDIAN(ItemContractQuantity). | Confirm quantity units vary and should be interpreted cautiously. |
| `QUANTITYP90` | quantity_stat | Yes | 90th percentile contract item quantity. | PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY ItemContractQuantity). | Confirm quantity units vary and should be interpreted cautiously. |
| `QUANTITYSTDDEV` | quantity_stat | Yes | Standard deviation of contract item quantities. | STDDEV(ItemContractQuantity). | Confirm quantity units vary and should be interpreted cautiously. |

## Budget Linkage

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `BUDGETPLANNEDVALUESUM` | currency | Maybe | Sum of linked budget item planned values. | SUM(COALESCE(BudgetItemUnitPrice * BudgetItemQuantity, 0)) across linked budget items. | Confirm linked budget values are setup-time and not revised after execution. |
| `HASANYLINKEDBUDGETITEMS` | flag | Maybe | Flag indicating at least one contract item links to a budget item. | IFF(COUNT(DISTINCT BudgetItemID) > 0, 1, 0). | Are budget links entered before project start and consistently across customers? |
| `NUMBUDGETMODULES` | count | Maybe | Number of distinct budget modules represented by linked budget items. | COUNT(DISTINCT BudgetModuleID). | Confirm module values and timing. |
| `SHAREITEMSLINKEDTOBUDGET` | share | Maybe | Fraction of contract items linked to budget items. | COUNT(DISTINCT ItemID where BudgetItemID IS NOT NULL) / NULLIF(NumContractItems, 0). | Confirm workflow timing and meaning of missing links. |
| `BUDGETTOITEMPLANNEDVALUERATIO` | ratio | Maybe | Ratio of linked budget planned value to contract item planned value. | BudgetPlannedValueSum / NULLIF(ProjectPlannedValue, 0). | Confirm whether signed project planned value is appropriate denominator. |

## Target/Post-Payment

| Field | Type | Beginning Available | Description | Derivation | Client Validation |
|---|---|---|---|---|---|
| `TARGETACTUALDURATIONDAYS` | target_component | No | Actual duration proxy from first posting minus 30 days to last posting. | DATEDIFF(day, DATEADD(day, -30, MIN(WPPostingDate)), MAX(WPPostingDate)). | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETACTUALENDDATE` | target_component | No | Actual project end proxy based on last valid posting date. | MAX(WPPostingDate) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETACTUALSTARTDATE` | target_component | No | Actual project start proxy based on first valid posting date minus 30 days. | DATEADD(day, -30, MIN(WPPostingDate)) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETFIRSTPOSTINGDATE` | target_component | No | First valid work posting date. | MIN(WPPostingDate) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETLASTPOSTINGDATE` | target_component | No | Last valid work posting date. | MAX(WPPostingDate) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETPAYMENTSPANDAYS` | target_component | No | Span in days from first to last valid posting date. | DATEDIFF(day, MIN(WPPostingDate), MAX(WPPostingDate)). | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETPLANNEDSTARTDATE` | target_component | No | Earliest contract start date among contracts/items that have valid posted payment rows. | MIN(ContractStartDate) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETVALIDPOSTEDDETAILROWS` | target_component | No | Count of valid posted payment detail rows. | COUNT(*) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETVALIDPOSTEDPAYDATECOUNT` | target_component | No | Count of distinct valid posting dates. | COUNT(DISTINCT WPPostingDate) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETVALIDPOSTEDPROJECTWORKCOMPLETEDAMOUNT` | target_component | No | Total calculated work completed amount on valid posted rows. | SUM(CalculatedWorkCompletedAmount) where CalculatedWorkCompletedAmount = CI.UnitPrice * PED.Quantity. | Use for retrospective target construction/audit only, not early prediction. |
| `PERCENTDELAYED` | target | No | Delay target used for modeling; positive means actual duration exceeded planned duration. | 100.0 * TargetActualDurationDays / NULLIF(TargetPlannedDurationDays, 0) - 100.0, with actual start as first posting date minus 30 days. | Confirm target definition and whether first posting minus 30 days is acceptable actual-start proxy. |
| `TARGETPLANNEDDURATIONDAYS` | target_component | No | Planned duration using only contracts/items with valid posted payment rows. | DATEDIFF(day, MIN(ContractStartDate), MAX(ContractClosureDate)) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETPLANNEDENDDATE` | target_component | No | Latest contract closure date among contracts/items that have valid posted payment rows. | MAX(ContractClosureDate) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |

## Full CSV Data Dictionary

The sortable/filterable CSV version is: `project_feature_non_keyword_field_dictionary_2026-06-10.csv`.