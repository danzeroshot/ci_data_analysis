# Original Project Feature Dataset Companion

Generated: 2026-06-11

## Purpose

This companion documents the original wide project-level feature dataset created before keyword review and column reduction. It describes the SQL lineage, dataset grain, source table assumptions, inclusion/filter behavior, target construction, keyword feature structure, and the non-keyword field dictionary.

## Dataset Lineage

- Generator SQL: `custpaydetails_project_feature_table_with_keywords_materialized.sql`
- Final Snowflake temporary table: `project_feature_table_with_keywords_final`
- Snowflake JSON export file: `custpaydetails_project_feature_table_with_keywords_materialized_2026-06-09-1526.csv`
- Flattened CSV used for analysis: `custpaydetails_project_feature_table_with_keywords_materialized_2026-06-09-1526_flat.csv`
- Later approved-keyword reduced derivative: `custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv`

The materialized SQL was a staged version of the optimized project feature query. It created temporary tables for base project/item setup rows, valid payment rows, target fields, text entities, item/contract/project aggregations, token rows, wide keyword features, and finally `project_feature_table_with_keywords_final`.

## Dataset Shape

- Parsed project rows: 5,762
- Total columns in flattened original feature dataset: 6,075
- Non-keyword columns: 75
- Keyword-derived columns: 6,000
- Grain: one row per `CustomerName + ProjectID`
- Customer sources: Amtrak, UDOT, CLV, CCD, Adams, Lincoln

Customer row counts in the flattened dataset:

| CustomerName | Rows |
| --- | --- |
| Lincoln | 3559 |
| UDOT | 1578 |
| CCD | 239 |
| CLV | 199 |
| Adams | 140 |
| Amtrak | 47 |

## Grain And Entity Assumptions

The project population is driven by contract item setup rows, not by payment rows. The base table `project_item_base` starts from each customer's `CORITEMItemDetails` rows where `CI.MODULEID = 'CONTMGT'`, joins to the contract master (`CONTMGTMaster`), project master (`PROJECTPROJECTMAIN`), project status lookup (`PROJECTProjectStatus`), optional linked budget item rows, and optional commitments.

A project is included when it has at least one contract item setup row joined to a project whose `PM.PROJECTNAME IS NOT NULL`. Because the final query starts from `project_nontext_features` and left-joins target and keyword features, projects are not required to have payment rows to appear in the final feature table.

The SQL uses `CustomerName + ProjectID` for the true project key. `ProjectName` is included as descriptive text, but it is not the uniqueness key by itself.

## Inclusion And Filtering Assumptions

The following filters and assumptions are embedded in the SQL:

- Base setup rows require `PM.PROJECTNAME IS NOT NULL`.
- Contract item setup rows require `CI.MODULEID = 'CONTMGT'` through the join condition.
- Budget item linkage is optional and limited to linked budget items whose module is in `('BDGTEST', 'BDGTREV')`.
- Commitment linkage is optional and limited to commitments where `C.POType = 'CONTMGT'` and `C.POTypeInstanceID = CM.ID`.
- Payment-derived target rows require `PM.PROJECTNAME IS NOT NULL`, `WP.STATUS IS NOT NULL`, `PE.STATUS IS NOT NULL`, and `WP.POSTINGDATE IS NOT NULL`.
- Payment rows are limited to work postings where `WP.ReferencePostingType = 'ITMPOST'`, `WP.PayItemID = CI.ItemID`, and `WP.POID = C.POID`.
- Payment detail rows are joined by `PED.WorkPostingID = WP.WPostingID` and `PED.WorkItemID = CI.ItemID`.
- No `ProjectStatus = Complete` filter is applied in this feature dataset.
- No minimum project value, minimum contract item count, or minimum payment count filter is applied in the final feature dataset.
- Keyword tokenization ignores text fragments shorter than three characters after lower-case/text-only normalization.

## Missing Data And Target Assumptions

Missing values are intentionally preserved in several places rather than dropping rows. This is important for modeling because some missingness is a data-quality signal and some is caused by target construction.

| Profile Item | Field | Rows | Share |
| --- | --- | --- | --- |
| Delay target populated | PERCENTDELAYED | 3,469 | 60.2% |
| Has first valid posting date | TARGETFIRSTPOSTINGDATE | 4,407 | 76.5% |
| Has last valid posting date | TARGETLASTPOSTINGDATE | 4,407 | 76.5% |
| Has valid posted detail rows | TARGETVALIDPOSTEDDETAILROWS | 4,407 | 76.5% |
| Planned duration populated | PLANNEDDURATIONDAYS | 4,266 | 74.0% |
| Has valid planned duration flag = 1 | HASVALIDPLANNEDDURATION | 4,106 | 71.3% |
| Project planned value populated | PROJECTPLANNEDVALUE | 5,762 | 100.0% |
| Project description populated | PROJECTDESCRIPTION | 4,960 | 86.1% |

Key target assumptions:

- `PercentDelayed` is retrospective and is not available at project start.
- `PercentDelayed = 100.0 * TargetActualDurationDays / TargetPlannedDurationDays - 100.0`.
- `TargetActualStartDate` is defined as first valid posting date minus 30 days.
- `TargetActualEndDate` is defined as the last valid posting date.
- `TargetPlannedStartDate` and `TargetPlannedEndDate` are computed from the subset of contracts/items that have valid posted payment rows, while `PlannedStartDate` and `PlannedEndDate` in the non-target feature set are computed from all setup contract entities in the project.
- `PercentDelayed` is null unless the target planned duration is positive and both planned/payment dates needed for the calculation exist.
- Projects without valid payment rows remain in the dataset, but target/payment fields are null.

## Implementation Note: Snowflake JSON Export

The final Snowflake table had roughly six thousand columns, which made direct `SELECT *` interaction and CSV download through the Snowflake web UI impractical. To export it, the final table was queried as JSON objects using an export wrapper equivalent to:

```sql
SELECT
    ROW_NUMBER() OVER (ORDER BY 1) AS RECORD_ID,
    OBJECT_CONSTRUCT(*) AS ALL_COLUMNS_JSON
FROM project_feature_table_with_keywords_final;
```

That JSON-shaped result was downloaded and then flattened locally into `custpaydetails_project_feature_table_with_keywords_materialized_2026-06-09-1526_flat.csv`. The `RECORD_ID` field is therefore an export artifact, not a business feature.

## Keyword Feature Structure

The original dataset includes the full pre-review keyword feature set. These columns were intentionally broad so downstream filtering could be performed outside Snowflake.

| Family | Source text | Keyword groups | Feature columns | Feature pattern |
| --- | --- | --- | --- | --- |
| Project text | PM.DESCRIPTION | 1,000 | 1,000 | PROJ_KW_<TOKEN>_COUNT |
| Contract text | CM.NAME + CM.DESC/Desc | 1,000 | 2,000 | CONTRACT_KW_<TOKEN>_CONTRACT_COUNT, CONTRACT_KW_<TOKEN>_CONTRACT_SHARE |
| Contract item text | CI.DESCRIPTION | 1,000 | 3,000 | ITEM_KW_<TOKEN>_ITEM_COUNT, ITEM_KW_<TOKEN>_ITEM_SHARE, ITEM_KW_<TOKEN>_PLANNED_VALUE_SHARE |

Keyword feature assumptions:

- Project keyword counts are based on project description text.
- Contract keyword counts and shares are based on contract name/description text at the contract level and aggregated to project level.
- Item keyword counts, item shares, and planned-value shares are based on contract item description text.
- Keyword features in this original dataset had not yet been filtered by the later min-4-project/multi-customer/manual semantic approval process.
- Several original keywords were later identified as customer/place-specific, generic administrative language, abbreviations, or noisy tokens. The approved reduced dataset should be preferred for customer-agnostic modeling.

## Beginning-Of-Project Availability Summary

Beginning-of-project means after project, contract, and contract item setup is entered, but before work postings, pay estimates, or payment history exists.

| Beginning Available | Field Count |
| --- | --- |
| Yes | 43 |
| No | 14 |
| Maybe | 6 |
| Yes, text field | 2 |
| Yes, if planned duration valid | 2 |
| Yes, if planned end date exists | 2 |
| Yes, but excluded for customer-agnostic model | 1 |
| Yes, but identifier/code-like | 1 |
| Yes, identifier only | 1 |
| Yes, if closure dates entered | 1 |
| Yes, if closure date is entered at setup | 1 |
| Maybe / generally not early-safe as currently extracted | 1 |

Field group counts:

| Field Group | Field Count |
| --- | --- |
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

## Modeling Guidance

Use these fields with care:

- Exclude `PERCENTDELAYED` from predictors; it is the outcome label.
- Exclude `TARGET*` fields from beginning-of-project prediction; they depend on payment history.
- Exclude `RECORD_ID`; it is only an export row number.
- Treat `CUSTOMERNAME` as available but intentionally excluded for customer-agnostic modeling.
- Treat `PROJECTSTATUS` cautiously; as extracted, it is the status at query time unless a status-history-as-of-prediction-date version is created.
- Treat budget linkage fields as potentially useful but workflow-dependent until the client confirms whether budget links are present before payment activity.
- Treat date fields from `CM.StartDt` and `CM.CLOSUREDT` as planned schedule inputs only if the client confirms they are not overwritten by actual schedule updates.
- Treat `UnitPrice`, `ContractQuantity`, and derived planned-value fields as beginning-available only if the client confirms they represent setup-time planned values or a stable current approved baseline.

## Non-Keyword Field Dictionary

The following table covers only the 75 non-keyword fields. Keyword columns are summarized by pattern above rather than enumerated individually.

| Field | Group | Type | Beginning Available | Missing Rate | Description | Derivation | Client Validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BUDGETPLANNEDVALUESUM` | budget linkage | currency | Maybe | 0.0% | Sum of linked budget item planned values. | SUM(COALESCE(BudgetItemUnitPrice * BudgetItemQuantity, 0)) across linked budget items. | Confirm linked budget values are setup-time and not revised after execution. |
| `BUDGETTOITEMPLANNEDVALUERATIO` | budget linkage | ratio | Maybe | 43.0% | Ratio of linked budget planned value to contract item planned value. | BudgetPlannedValueSum / NULLIF(ProjectPlannedValue, 0). | Confirm whether signed project planned value is appropriate denominator. |
| `HASANYLINKEDBUDGETITEMS` | budget linkage | flag | Maybe | 0.0% | Flag indicating at least one contract item links to a budget item. | IFF(COUNT(DISTINCT BudgetItemID) > 0, 1, 0). | Are budget links entered before project start and consistently across customers? |
| `NUMBUDGETMODULES` | budget linkage | count | Maybe | 0.0% | Number of distinct budget modules represented by linked budget items. | COUNT(DISTINCT BudgetModuleID). | Confirm module values and timing. |
| `SHAREITEMSLINKEDTOBUDGET` | budget linkage | share | Maybe | 0.0% | Fraction of contract items linked to budget items. | COUNT(DISTINCT ItemID where BudgetItemID IS NOT NULL) / NULLIF(NumContractItems, 0). | Confirm workflow timing and meaning of missing links. |
| `CUSTOMERNAME` | identity/export | categorical | Yes, but excluded for customer-agnostic model | 0.0% | Customer/source environment name: Amtrak, UDOT, CLV, CCD, Adams, or Lincoln. | Hardcoded per source schema branch in SQL. | Confirm whether future production use should remain customer-agnostic or allow customer-specific calibration. |
| `RECORD_ID` | identity/export | identifier | No | 0.0% | Sequential row number added in the Snowflake JSON export query, not a business field. | ROW_NUMBER() OVER (ORDER BY 1) in export wrapper. | None; exclude from modeling. |
| `PROJECTCODE` | identity/project | text/code | Yes, but identifier/code-like | 0.0% | Project code. | PM.ProjectCode aggregated as MAX at project grain. | Does project code encode schedule, funding, geography, or other future/outcome information? |
| `PROJECTDESCRIPTION` | identity/project | text | Yes, text field | 13.9% | Project description. | PM.Description aggregated as MAX at project grain. | Confirm description is stable and available before payments. |
| `PROJECTID` | identity/project | identifier | Yes, identifier only | 0.0% | Project primary key from source project table. | PM.ProjectID. | Use only for joins/audits, not modeling. |
| `PROJECTNAME` | identity/project | text | Yes, text field | 0.0% | Project name. | PM.ProjectName aggregated as MAX at project grain. | Confirm project name is entered before contract item setup. |
| `QUANTITYMAX` | item quantity | quantity_stat | Yes | 0.0% | Maximum contract item quantity. | MAX(ItemContractQuantity). | Confirm quantity outliers are meaningful. |
| `QUANTITYMEDIAN` | item quantity | quantity_stat | Yes | 0.0% | Median contract item quantity. | MEDIAN(ItemContractQuantity). | Confirm quantity units vary and should be interpreted cautiously. |
| `QUANTITYP90` | item quantity | quantity_stat | Yes | 0.0% | 90th percentile contract item quantity. | PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY ItemContractQuantity). | Confirm quantity units vary and should be interpreted cautiously. |
| `QUANTITYSTDDEV` | item quantity | quantity_stat | Yes | 2.1% | Standard deviation of contract item quantities. | STDDEV(ItemContractQuantity). | Confirm quantity units vary and should be interpreted cautiously. |
| `UNITPRICEMAX` | item unit price | currency_stat | Yes | 0.0% | Maximum item unit price across contract items. | MAX(ItemUnitPrice). | Confirm unit price outliers are meaningful. |
| `UNITPRICEMEDIAN` | item unit price | currency_stat | Yes | 0.0% | Median item unit price across contract items. | MEDIAN(ItemUnitPrice). | Confirm unit price is original setup value, not revised current value. |
| `UNITPRICEP90` | item unit price | currency_stat | Yes | 0.0% | 90th percentile item unit price across contract items. | PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY ItemUnitPrice). | Confirm unit price outliers are meaningful. |
| `UNITPRICESTDDEV` | item unit price | currency_stat | Yes | 2.1% | Standard deviation of item unit prices. | STDDEV(ItemUnitPrice). | Confirm mixing different units makes this feature interpretable enough. |
| `ABSPROJECTPLANNEDVALUE` | planned money | currency | Yes | 0.0% | Sum of absolute planned contract item values for the project. | SUM(ABS(COALESCE(ItemUnitPrice * ItemContractQuantity, 0))) across project items. | Confirm negative item values are adjustments and should contribute by absolute value. |
| `LOG1PABSPROJECTPLANNEDVALUE` | planned money | numeric_transform | Yes | 0.0% | Log transform of absolute project planned value. | LN(1 + ABS(AbsProjectPlannedValue)). | None. |
| `PROJECTPLANNEDVALUE` | planned money | currency | Yes | 0.0% | Sum of planned contract item values for the project; signed. | SUM(COALESCE(ItemUnitPrice * ItemContractQuantity, 0)) across project items. | Confirm item unit price and quantity represent original planned value at setup and are not overwritten later. |
| `SHARENEGATIVEPLANNEDVALUEITEMS` | planned money/data quality | share | Yes | 0.0% | Fraction of contract items with negative planned value. | COUNT(DISTINCT ItemID where ItemPlannedValue < 0) / NULLIF(NumContractItems, 0). | Ask client what negative planned-value items represent. |
| `SHAREZEROPLANNEDVALUEITEMS` | planned money/data quality | share | Yes | 0.0% | Fraction of contract items with zero planned value. | COUNT(DISTINCT ItemID where ItemPlannedValue = 0) / NULLIF(NumContractItems, 0). | Ask client what zero planned-value items represent. |
| `ITEMPLANNEDVALUEHERFINDAHL` | planned money/item concentration | concentration_index | Yes | 41.6% | Herfindahl concentration index of absolute planned item values. | SUM(POWER(ABS(ItemPlannedValue) / NULLIF(AbsProjectPlannedValue, 0), 2)). | Confirm absolute-value concentration is appropriate. |
| `MAXITEMSHAREOFPLANNEDVALUE` | planned money/item concentration | share | Yes | 41.6% | Largest absolute item value as a share of total absolute planned value. | MAX(ABS(ItemPlannedValue) / NULLIF(AbsProjectPlannedValue, 0)). | Confirm concentration is meaningful and absolute value is preferred. |
| `MAXITEMPLANNEDVALUE` | planned money/item distribution | currency_stat | Yes | 0.0% | Largest signed planned item value within the project. | MAX(ItemUnitPrice * ItemContractQuantity). | Confirm signed max is desired; negative-only projects may behave oddly. |
| `MEANITEMPLANNEDVALUE` | planned money/item distribution | currency_stat | Yes | 0.0% | Mean planned value of contract items. | AVG(ItemUnitPrice * ItemContractQuantity). | Confirm current item values are original planned values. |
| `MEDIANITEMPLANNEDVALUE` | planned money/item distribution | currency_stat | Yes | 0.0% | Median planned value of contract items. | MEDIAN(ItemUnitPrice * ItemContractQuantity). | Confirm current item values are original planned values. |
| `STDDEVITEMPLANNEDVALUE` | planned money/item distribution | currency_stat | Yes | 2.1% | Standard deviation of planned item values. | STDDEV(ItemUnitPrice * ItemContractQuantity). | Confirm outliers are real scope values vs data issues. |
| `DOLLARSPERPLANNEDDAY` | planned money/schedule | currency_rate | Yes, if planned duration valid | 28.1% | Planned project value divided by planned duration in days. | ProjectPlannedValue / NULLIF(PlannedDurationDays, 0). | Confirm signed planned value should be used vs absolute planned value. |
| `DOLLARSPERPLANNEDMONTH` | planned money/schedule | currency_rate | Yes, if planned duration valid | 28.1% | Planned project value divided by planned duration in approximate months. | ProjectPlannedValue / NULLIF(PlannedDurationDays / 30.0, 0). | Confirm 30-day month convention. |
| `DOLLARSPERCONTRACT` | planned money/scope | currency_rate | Yes | 0.0% | Planned project value per contract. | ProjectPlannedValue / NULLIF(NumContracts, 0). | Confirm signed planned value should be used vs absolute planned value. |
| `DOLLARSPERCONTRACTITEM` | planned money/scope | currency_rate | Yes | 0.0% | Planned project value per contract item. | ProjectPlannedValue / NULLIF(NumContractItems, 0). | Confirm signed planned value should be used vs absolute planned value. |
| `CONTRACTENDSPREADDAYS` | planned schedule | duration_days | Yes, if closure dates entered | 26.0% | Spread in planned contract end/closure dates within project. | DATEDIFF(day, MIN(ContractClosureDate), MAX(ContractClosureDate)). | Confirm closure dates are planned dates and not actual closeout dates. |
| `CONTRACTSTARTSPREADDAYS` | planned schedule | duration_days | Yes | 0.2% | Spread in planned contract start dates within project. | DATEDIFF(day, MIN(ContractStartDate), MAX(ContractStartDate)). | Confirm multiple-contract projects are expected and contract start dates are planned dates. |
| `HASPLANNEDENDDATE` | planned schedule | flag | Yes | 0.0% | Flag indicating the project has at least one planned end/closure date. | IFF(MAX(ContractClosureDate) IS NOT NULL, 1, 0). | Confirm missing closure date indicates incomplete setup vs valid open-ended project. |
| `HASVALIDPLANNEDDURATION` | planned schedule | flag | Yes | 0.0% | Flag indicating planned start and end exist and planned duration is positive. | IFF(MIN(ContractStartDate) IS NOT NULL AND MAX(ContractClosureDate) IS NOT NULL AND DATEDIFF(day, MIN(start), MAX(end)) > 0, 1, 0). | Confirm nonpositive planned durations should be treated as invalid setup. |
| `LOG1PPLANNEDDURATIONDAYS` | planned schedule | numeric_transform | Yes | 26.0% | Log transform of nonnegative planned duration. | LN(1 + GREATEST(PlannedDurationDays, 0)). | Confirm whether negative planned durations should be nulled instead of clipped to zero for log transform. |
| `MAXCONTRACTPLANNEDDURATIONDAYS` | planned schedule | duration_days_stat | Yes | 26.0% | Maximum planned contract duration within project. | MAX(DATEDIFF(day, ContractStartDate, ContractClosureDate)) for contracts with non-null dates. | Confirm outlier contract durations are meaningful. |
| `MEDIANCONTRACTPLANNEDDURATIONDAYS` | planned schedule | duration_days_stat | Yes | 26.0% | Median planned duration across contracts in the project. | MEDIAN(DATEDIFF(day, ContractStartDate, ContractClosureDate)) for contracts with non-null dates. | Confirm contract-level planned duration definition. |
| `NUMCONTRACTSWITHINVALIDSCHEDULE` | planned schedule | count | Yes | 0.0% | Number of contracts with missing or nonpositive planned schedule. | COUNT(DISTINCT ContractID where start is null OR end is null OR DATEDIFF(day,start,end)<=0). | Confirm invalid schedule interpretation. |
| `NUMCONTRACTSWITHVALIDSCHEDULE` | planned schedule | count | Yes | 0.0% | Number of contracts with non-null start/end dates and positive duration. | COUNT(DISTINCT ContractID where start/end not null and DATEDIFF(day,start,end)>0). | Confirm invalid schedules reflect data quality/setup risk. |
| `PLANNEDDURATIONDAYS` | planned schedule | duration_days | Yes | 26.0% | Planned project duration in days from earliest contract start to latest contract closure. | DATEDIFF(day, MIN(ContractStartDate), MAX(ContractClosureDate)). | Confirm desired planned duration definition: project-level contract envelope vs official PM project dates. |
| `PLANNEDENDDATE` | planned schedule | date | Yes, if closure date is entered at setup | 26.0% | Latest planned contract closure/end date across contracts in the project. | MAX(ContractClosureDate) across distinct contracts for the project. | Confirm CM.ClosureDt is planned end date and whether it changes during execution. |
| `PLANNEDENDMONTH` | planned schedule | calendar | Yes, if planned end date exists | 26.0% | Calendar month of planned project end. | EXTRACT(month FROM MAX(ContractClosureDate)). | Confirm planned end date is available before work starts. |
| `PLANNEDENDQUARTER` | planned schedule | calendar | Yes, if planned end date exists | 26.0% | Calendar quarter of planned project end. | EXTRACT(quarter FROM MAX(ContractClosureDate)). | Confirm planned end date is available before work starts. |
| `PLANNEDSTARTDATE` | planned schedule | date | Yes | 0.2% | Earliest planned contract start date across contracts in the project. | MIN(ContractStartDate) across distinct contracts for the project. | Confirm CM.StartDt is planned/original start and not overwritten later. |
| `PLANNEDSTARTMONTH` | planned schedule | calendar | Yes | 0.2% | Calendar month of planned project start. | EXTRACT(month FROM MIN(ContractStartDate)). | Confirm month seasonality is acceptable for modeling. |
| `PLANNEDSTARTQUARTER` | planned schedule | calendar | Yes | 0.2% | Calendar quarter of planned project start. | EXTRACT(quarter FROM MIN(ContractStartDate)). | Confirm quarter seasonality is acceptable for modeling. |
| `SHARECONTRACTSWITHINVALIDSCHEDULE` | planned schedule | share | Yes | 0.0% | Fraction of contracts with missing or invalid planned schedule. | NumContractsWithInvalidSchedule / NULLIF(NumContracts, 0). | Confirm invalid schedule interpretation. |
| `STDDEVCONTRACTPLANNEDDURATIONDAYS` | planned schedule | duration_days_stat | Yes | 97.6% | Standard deviation of planned contract durations within project. | STDDEV(DATEDIFF(day, ContractStartDate, ContractClosureDate)) for contracts with non-null dates. | Confirm dispersion across contracts is meaningful. |
| `ITEMSPERCONTAINER` | scope complexity | ratio | Yes | 0.1% | Average number of contract items per item container. | COUNT(DISTINCT ItemID) / NULLIF(COUNT(DISTINCT ItemContainerID), 0). | None. |
| `ITEMSPERCONTRACT` | scope complexity | ratio | Yes | 0.0% | Average number of contract items per contract. | NumContractItems / NULLIF(NumContracts, 0). | None. |
| `LOG1PNUMCONTRACTITEMS` | scope complexity | numeric_transform | Yes | 0.0% | Log transform of contract item count. | LN(1 + COUNT(DISTINCT ItemID)). | None. |
| `NUMCOMMITMENTS` | scope complexity | count | Maybe | 0.0% | Number of distinct commitment purchase/order IDs linked to contracts. | COUNT(DISTINCT CommitmentPOID). | Are commitments always created before any work postings/payments? |
| `NUMCONTRACTITEMS` | scope complexity | count | Yes | 0.0% | Number of distinct contract items on the project. | COUNT(DISTINCT ItemID). | Confirm prediction point is after contract items are entered. |
| `NUMCONTRACTS` | scope complexity | count | Yes | 0.0% | Number of distinct contracts on the project. | COUNT(DISTINCT ContractID). | Confirm prediction point is after all initial contracts are entered. |
| `NUMDISTINCTSTANDARDITEMPREFIX3` | scope complexity | count | Yes | 0.0% | Number of distinct 3-character prefixes from standard item numbers. | COUNT(DISTINCT REGEXP_SUBSTR(StandardItemNo, ^[A-Za-z0-9]{3})). | Confirm prefix length is meaningful across customers. |
| `NUMDISTINCTSTANDARDITEMPREFIX5` | scope complexity | count | Yes | 0.0% | Number of distinct 5-character prefixes from standard item numbers. | COUNT(DISTINCT REGEXP_SUBSTR(StandardItemNo, ^[A-Za-z0-9]{5})). | Confirm prefix length is meaningful across customers. |
| `NUMITEMCONTAINERS` | scope complexity | count | Yes | 0.0% | Number of distinct item containers/work-breakdown containers used by project items. | COUNT(DISTINCT ItemContainerID). | Confirm container structure is stable at setup. |
| `PERCENTDELAYED` | target/post-payment | target | No | 39.8% | Delay target used for modeling; positive means actual duration exceeded planned duration. | 100.0 * TargetActualDurationDays / NULLIF(TargetPlannedDurationDays, 0) - 100.0, with actual start as first posting date minus 30 days. | Confirm target definition and whether first posting minus 30 days is acceptable actual-start proxy. |
| `TARGETACTUALDURATIONDAYS` | target/post-payment | target_component | No | 23.5% | Actual duration proxy from first posting minus 30 days to last posting. | DATEDIFF(day, DATEADD(day, -30, MIN(WPPostingDate)), MAX(WPPostingDate)). | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETACTUALENDDATE` | target/post-payment | target_component | No | 23.5% | Actual project end proxy based on last valid posting date. | MAX(WPPostingDate) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETACTUALSTARTDATE` | target/post-payment | target_component | No | 23.5% | Actual project start proxy based on first valid posting date minus 30 days. | DATEADD(day, -30, MIN(WPPostingDate)) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETFIRSTPOSTINGDATE` | target/post-payment | target_component | No | 23.5% | First valid work posting date. | MIN(WPPostingDate) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETLASTPOSTINGDATE` | target/post-payment | target_component | No | 23.5% | Last valid work posting date. | MAX(WPPostingDate) from valid payment rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETPAYMENTSPANDAYS` | target/post-payment | target_component | No | 23.5% | Span in days from first to last valid posting date. | DATEDIFF(day, MIN(WPPostingDate), MAX(WPPostingDate)). | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETPLANNEDDURATIONDAYS` | target/post-payment | target_component | No | 38.3% | Planned duration using only contracts/items with valid posted payment rows. | DATEDIFF(day, MIN(ContractStartDate), MAX(ContractClosureDate)) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETPLANNEDENDDATE` | target/post-payment | target_component | No | 38.3% | Latest contract closure date among contracts/items that have valid posted payment rows. | MAX(ContractClosureDate) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETPLANNEDSTARTDATE` | target/post-payment | target_component | No | 23.5% | Earliest contract start date among contracts/items that have valid posted payment rows. | MIN(ContractStartDate) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETVALIDPOSTEDDETAILROWS` | target/post-payment | target_component | No | 23.5% | Count of valid posted payment detail rows. | COUNT(*) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETVALIDPOSTEDPAYDATECOUNT` | target/post-payment | target_component | No | 23.5% | Count of distinct valid posting dates. | COUNT(DISTINCT WPPostingDate) from payment_rows. | Use for retrospective target construction/audit only, not early prediction. |
| `TARGETVALIDPOSTEDPROJECTWORKCOMPLETEDAMOUNT` | target/post-payment | target_component | No | 23.5% | Total calculated work completed amount on valid posted rows. | SUM(CalculatedWorkCompletedAmount) where CalculatedWorkCompletedAmount = CI.UnitPrice * PED.Quantity. | Use for retrospective target construction/audit only, not early prediction. |
| `PROJECTSTATUS` | workflow/status | categorical | Maybe / generally not early-safe as currently extracted | 1.0% | Current project status name from status lookup. | LPS.StatusName joined from PM.StatusId and aggregated as MAX. | Can status history be reconstructed as of prediction date? If not, exclude from early model. |
