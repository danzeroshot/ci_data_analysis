# Project Delay Random Forest Pct/Bucket Keyword Summary

- The revised pct/bucket dataset contains 3,331 projects and 546 columns.
- all: adding bucket-eligible selection changed regression R2 by -0.001, MAE by -0.07, and classification AUC by -0.000.
- usable: adding bucket-eligible selection changed regression R2 by -0.004, MAE by 0.25, and classification AUC by 0.000.
- Interpretation: positive R2/AUC deltas and negative MAE deltas indicate that bucket features or bucket-influenced feature selection added value.
- The usable-only scenarios are the better approximation of a deployable early-risk model; the all-field scenarios can still include retrospective payment-history signals.
