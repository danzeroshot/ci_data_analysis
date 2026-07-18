# Project Delay Correlation Analysis Summary

- The corrected dataset contains 3,331 project rows, 394 columns, and 300 keyword predictor columns.
- Strongest candidate Spearman correlation with PERCENTDELAYED: PAYDATESPERPLANNEDMONTH (numeric predictor), rho=0.854.
- Strongest positive candidate: PAYDATESPERPLANNEDMONTH, rho=0.854.
- Strongest negative candidate: SHAREMISSINGORZEROPLANNEDVALUEITEMS, rho=-0.758.
- Strongest keyword candidate: CI_KW_MOBILIZATION (keyword: CI.DESCRIPTION), rho=0.747.
- Fields derived from actual payment timing are flagged as leakage: useful for retrospective explanation, but not valid as forward-looking predictors.
- Next step: rerun the same correlation screen within customer or customer/type strata to see whether pooled correlations are dominated by customer mix.
