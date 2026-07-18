# Approved Keyword Feature Dataset Process

Date: 2026-06-11

Output feature dataset: `custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv`
Column-level filter detail: `approved_keyword_feature_column_filter_detail_2026-06-11.csv`

## Purpose

This dataset starts from the full project-level feature table and removes keyword features that are too sparse, customer-specific, or manually reviewed as low-value/noisy. All non-keyword project fields are retained.

## Starting Point

- Source flat feature CSV: `custpaydetails_project_feature_table_with_keywords_materialized_2026-06-09-1526_flat.csv`
- Source rows: 5,762
- Original columns: 6,075
- Non-keyword columns: 75
- Original keyword candidate groups: 3,000 total
  - 1,000 project keyword groups
  - 1,000 contract keyword groups
  - 1,000 item keyword groups
- Original keyword feature columns: 6,000
  - project: 1 feature per keyword group
  - contract: count + share, 2 features per keyword group
  - item: count + share + planned-value share, 3 features per keyword group

## Automated Frequency / Generalization Filter

A keyword family row was retained by rule only if its count column was positive in at least 4 unique project rows and appeared across more than 1 customer.

| Stage | Keyword Family Rows |
|---|---:|
| Original project/contract/item keyword family rows | 3,000 |
| Retained after min-4-projects and multi-customer rule | 2,069 |
| Dropped by automated rule | 931 |

Retained by family after automated rule:

| Family | Retained Family Rows |
|---|---:|
| project | 664 |
| contract | 521 |
| item | 884 |

## Manual Review Filter

The retained unique-keyword review file was `retained_keywords_unique_min4_projects_multicustomer_review_2026-06-11.csv`. A unique keyword was removed when `recommended_remove_keyword = TRUE`. If a unique keyword was removed, all of its retained family-specific feature columns were excluded from this approved feature dataset.

| Manual Review Result | Unique Keywords |
|---|---:|
| Unique retained keywords reviewed | 1,361 |
| Approved unique keywords | 1,276 |
| Recommended removal unique keywords | 85 |

Removal count by original review category:

| Original Review Category | Removed Unique Keywords |
|---|---:|
| Retained but likely lower semantic value: generic administrative/project word. Keep only if model validation shows signal or client confirms consistent business meaning. | 31 |
| Retained but review carefully: likely geography/customer/place-specific. It passes support/customer-span criteria, but may encode location rather than transferable project scope. | 26 |
| Retained in at least one family but dropped in another; value depends on which text source is trusted. Review family-specific meaning before final use. | 23 |
| Retained but needs client review: abbreviation/code/unit-like token. Could be meaningful domain shorthand or noise. | 5 |

Manual review removed retained family rows:

| Stage | Keyword Family Rows |
|---|---:|
| Retained after automated rule | 2,069 |
| Removed by manual review | 129 |
| Approved retained keyword family rows | 1,940 |

Approved retained family rows by family:

| Family | Approved Family Rows |
|---|---:|
| project | 601 |
| contract | 477 |
| item | 862 |

## Final Dataset Shape

| Measure | Count |
|---|---:|
| Rows | 5,762 |
| Total output columns | 4,216 |
| Non-keyword columns retained | 75 |
| Approved keyword feature columns retained | 4,141 |
| Columns removed from original full dataset | 1,859 |

## Important Interpretation Notes

- The manual review is intentionally conservative for customer-agnostic modeling.
- Proper place names, customer names, and street/subdivision names were generally removed unless they appeared to represent transferable physical context, such as waterways or terrain.
- Abbreviations were kept when they appeared to be meaningful infrastructure/domain shorthand, such as material, environmental, utility, lighting, or pavement acronyms.
- Generic administrative words were removed when source-context review showed mostly boilerplate or unstable standalone meaning.
- The companion column-level filter detail CSV lists every original column and whether it was retained in the approved output.