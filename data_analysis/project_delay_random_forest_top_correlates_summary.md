# Project Delay Random Forest Summary

- The all-top-correlate model uses 40 fields, including retrospective payment-history fields.
- The usable-only model uses 23 fields after removing 17 payment-history-dependent fields.
- Regression R2 changed from 0.831 to 0.465 after removing non-usable fields.
- Regression MAE changed from 29.39 to 98.50.
- Classification ROC AUC changed from 0.990 to 0.970.
- If the all-top model is materially stronger, that advantage should be treated as diagnostic rather than deployable because it can rely on payment cadence, burn, and spread fields that are only known after work has progressed.
- The usable-only model is the better proxy for whether project metadata, contract/item setup, and text fields can support early delay risk scoring.
