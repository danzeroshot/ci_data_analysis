# Approved Keyword Feature Correlation Source Mapping

Dataset: `custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv`
Target: `PERCENTDELAYED`

This extract ranks non-leakage numeric features by Spearman correlation against project percent delay. Positive and negative features are ranked separately, matching the prior top-correlates notebook convention. An absolute rank is also included for a single overall strength ordering.

Excluded fields were written to a separate CSV with the same schema. Exclusions include the target itself, retrospective target/leakage fields, identity/descriptive fields, and nonnumeric fields where Spearman correlation is not meaningful.

## Counts

- Primary ranked features: 4,194
- Positive Spearman features: 2,773
- Negative Spearman features: 1,349
- Zero/undefined Spearman features: 72
- Excluded fields: 22

## Top Positive Features

| direction_rank | feature | spearman_r | class |
| --- | --- | --- | --- |
| 1 | DOLLARSPERPLANNEDDAY | 0.765886 | planned money/schedule |
| 2 | DOLLARSPERPLANNEDMONTH | 0.765886 | planned money/schedule |
| 3 | ITEM_KW_MOBILIZATION_ITEM_COUNT | 0.723416 | item_keyword_count |
| 4 | UNITPRICEMEDIAN | 0.713655 | item unit price |
| 5 | UNITPRICEP90 | 0.713235 | item unit price |
| 6 | UNITPRICEMAX | 0.704343 | item unit price |
| 7 | UNITPRICESTDDEV | 0.700733 | item unit price |
| 8 | ITEM_KW_MOBILIZATION_ITEM_SHARE | 0.687811 | item_keyword_share |
| 9 | ITEM_KW_TRAFFIC_ITEM_COUNT | 0.673954 | item_keyword_count |
| 10 | DOLLARSPERCONTRACT | 0.670817 | planned money/scope |

## Top Negative Features

| direction_rank | feature | spearman_r | class |
| --- | --- | --- | --- |
| 1 | MEDIANCONTRACTPLANNEDDURATIONDAYS | -0.727600 | planned schedule |
| 2 | MAXCONTRACTPLANNEDDURATIONDAYS | -0.723721 | planned schedule |
| 3 | LOG1PPLANNEDDURATIONDAYS | -0.721927 | planned schedule |
| 4 | PLANNEDDURATIONDAYS | -0.721927 | planned schedule |
| 5 | SHAREZEROPLANNEDVALUEITEMS | -0.697997 | planned money/data quality |
| 6 | CONTRACT_KW_ADDITION_CONTRACT_COUNT | -0.634727 | contract_keyword_count |
| 7 | CONTRACT_KW_ADDITION_CONTRACT_SHARE | -0.634727 | contract_keyword_share |
| 8 | PROJ_KW_ADDITION_COUNT | -0.634180 | project_keyword_count |
| 9 | STDDEVCONTRACTPLANNEDDURATIONDAYS | -0.612174 | planned schedule |
| 10 | PROJ_KW_SEWER_COUNT | -0.469554 | project_keyword_count |

## Source Mapping Notes

- Non-keyword features use the derivation documented in the project feature field dictionary.
- Project keyword features are token counts from `PROJECTPROJECTMAIN.DESCRIPTION`.
- Contract keyword features are token counts or shares from `CONTMGTMASTER.NAME` and `CONTMGTMASTER.DESC`.
- Item keyword features are token counts, item shares, or planned-value shares from `CORITEMITEMDETAILS.DESCRIPTION` and planned item values.

## Outputs

- Primary table: `approved_keyword_feature_spearman_correlations_with_sources_2026-06-15.csv`
- Excluded table: `approved_keyword_feature_spearman_correlations_excluded_features_2026-06-15.csv`
