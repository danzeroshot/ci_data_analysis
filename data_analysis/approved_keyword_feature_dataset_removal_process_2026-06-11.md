# Approved Keyword Feature Dataset Removal Process

Generated: 2026-06-11

## Output Dataset

- Source flat feature table: `custpaydetails_project_feature_table_with_keywords_materialized_2026-06-09-1526_flat.csv`
- Reviewed retained keyword file: `retained_keywords_unique_min4_projects_multicustomer_review_2026-06-11.csv`
- New approved feature dataset: `custpaydetails_project_feature_table_with_approved_keywords_2026-06-11.csv`
- Parsed project rows: 5,762
- Output columns: 4,216
- Non-keyword columns retained unchanged: 75
- Approved keyword feature columns retained: 4,141

The output keeps every non-keyword project feature from the flat project-level feature table. Keyword-derived feature columns are retained only when the keyword family passed the statistical screen and the keyword was not manually flagged for removal in the retained keyword review file.

## Original Keyword Feature Set

The original materialized project feature table contained 3,000 keyword groups across three text families:

- Project description keywords: 1,000 groups, represented by 1 feature column per keyword.
- Contract name/description keywords: 1,000 groups, represented by 2 feature columns per keyword.
- Contract item description keywords: 1,000 groups, represented by 3 feature columns per keyword.

That produced 6,000 keyword-derived feature columns, plus 75 non-keyword project/date/money/status/count fields.

## Filtering And Review Process

1. Exploratory minimum total count screen: keyword families were first evaluated with a minimum total usage threshold of 4. This retained 2,770 of 3,000 family keyword groups, but it was not strict enough because several terms were repeated within a small number of projects.

2. Unique project usage screen: the threshold was tightened to require use in at least 4 unique projects. This retained 2,708 of 3,000 family keyword groups.

3. Multi-customer screen: the final statistical screen required each retained family keyword group to appear in at least 4 unique projects and more than 1 customer. This retained 2,069 family keyword groups and 4,358 keyword feature columns before manual review.

4. Unique keyword consolidation: the retained family keyword groups represented 1,361 unique keyword strings. A keyword could be retained in one text family while failing the screen in another; retention was tracked at the keyword-family level.

5. Manual/context review: the retained unique keywords were reviewed for semantic usefulness. The review flagged 85 unique retained keywords for removal and approved 1,276 unique retained keywords for modeling use.

## Manual Removal Summary

The retained keyword review file contains the line-by-line rationale in the semantic review and removal columns. Removal counts by recorded reason are:

- Generic-admin context review did not find clear standalone semantic value; remove from customer-agnostic keyword feature set unless client overrides.: 31
- Geography/place context review recommends removal: Likely place/subdivision/project name; geography proxy.: 2
- Geography/place context review recommends removal: Likely subdivision/development/place name; geography proxy.: 2
- Geography/place context review recommends removal: Proper place name; geography proxy.: 2
- Mixed-family context review recommends removal: Likely place/subdivision name; geography proxy.: 2
- Abbreviation/code context review recommends removal: COR appears as change-order/request/admin shorthand in item text; too process-specific/noisy for general beginning keyword feature.: 1
- Abbreviation/code context review recommends removal: PKWY is parkway/street-name abbreviation; primarily location/name proxy rather than scope.: 1
- Abbreviation/code context review recommends removal: RFI is a construction administration process token; likely post-start/process-related and not stable physical scope.: 1
- Abbreviation/code context review recommends removal: STA appears to be stationing/location notation; very common but weak standalone scope meaning.: 1
- Abbreviation/code context review recommends removal: TYP is typical notation; drafting/spec notation with little standalone semantic value.: 1
- Geography/place context review recommends removal: Customer/city/place name; likely encodes source/customer geography.: 1
- Geography/place context review recommends removal: Likely corridor/place name in sampled project text; not a transferable scope term by itself.: 1
- Geography/place context review recommends removal: Likely facility/place/project name; geography proxy.: 1
- Geography/place context review recommends removal: Likely place/subdivision/road name; geography proxy.: 1
- Geography/place context review recommends removal: Likely road/place/facility name; geography proxy.: 1
- Geography/place context review recommends removal: Likely street/place name; geography proxy.: 1
- Geography/place context review recommends removal: Likely subdivision/development/place-name token; not transferable scope.: 1
- Geography/place context review recommends removal: Mostly New York/place-name context; geography/customer proxy.: 1
- Geography/place context review recommends removal: Mostly part of Salt Lake/place names; geography proxy unless client confirms material usage.: 1
- Geography/place context review recommends removal: Mostly proper place/facility/corridor name; weak transferable scope signal.: 1
- Geography/place context review recommends removal: Part of Las Vegas/name text; geography proxy and fragmentary token.: 1
- Geography/place context review recommends removal: Proper city/place name; geography/customer proxy.: 1
- Geography/place context review recommends removal: Proper place name; geography/customer proxy.: 1
- Geography/place context review recommends removal: Proper place/corridor name; geography proxy.: 1
- Geography/place context review recommends removal: Proper place/customer/county name; likely encodes geography/customer rather than transferable project scope.: 1
- Geography/place context review recommends removal: Proper place/street name; geography proxy.: 1
- Geography/place context review recommends removal: Proper place/street/person name; geography proxy.: 1
- Geography/place context review recommends removal: Proper road/person/place name; geography proxy.: 1
- Geography/place context review recommends removal: Proper road/place name; geography proxy.: 1
- Geography/place context review recommends removal: Proper street/place/person name; geography proxy.: 1
- Mixed-family context review recommends removal: Ambiguous as verb/name/marking fragment; weaker than explicit marking/striping terms.: 1
- Mixed-family context review recommends removal: Appears to be place/name-specific in sampled contexts; likely geography/customer vocabulary.: 1
- Mixed-family context review recommends removal: Directional/geographic term; likely location proxy.: 1
- Mixed-family context review recommends removal: Funding/jurisdiction descriptor; may encode program/customer context more than scope.: 1
- Mixed-family context review recommends removal: Generic adjective; weak standalone signal.: 1
- Mixed-family context review recommends removal: Generic narrative word; no standalone scope value.: 1
- Mixed-family context review recommends removal: Generic project verb; weak standalone scope signal.: 1
- Mixed-family context review recommends removal: Generic versioning/reference word; weak standalone value.: 1
- Mixed-family context review recommends removal: Generic/noisy token; weak standalone semantic meaning.: 1
- Mixed-family context review recommends removal: Geographic/organizational descriptor; likely customer/geography proxy.: 1
- Mixed-family context review recommends removal: Likely place/project name; geography proxy.: 1
- Mixed-family context review recommends removal: Likely school/place-specific project naming; not transferable scope by itself.: 1
- Mixed-family context review recommends removal: Likely status/outcome or generic closure language; not stable beginning scope signal.: 1
- Mixed-family context review recommends removal: Likely subdivision/place/project-name term; geography proxy.: 1
- Mixed-family context review recommends removal: Mixed usage as geography/river bank/project name; too ambiguous for customer-agnostic keyword feature.: 1
- Mixed-family context review recommends removal: Mostly organizational/geographic label; weak scope meaning.: 1
- Mixed-family context review recommends removal: Mostly planning/analysis language rather than construction scope; context is broad and brittle across text sources.: 1
- Mixed-family context review recommends removal: Mostly study/assessment language; broad service/planning word with limited standalone scope meaning.: 1
- Mixed-family context review recommends removal: Often street/place geometry/name; ambiguous and likely location-specific.: 1
- Mixed-family context review recommends removal: Sampled usage is too ambiguous/noisy as standalone; could mean counter item, counting, or accounting context.: 1
- Mixed-family context review recommends removal: Stopword-like verb; no standalone semantic value.: 1

The major review buckets were:

- Generic administrative/project words: broad words were checked against their source contexts and removed when they did not add useful semantic distinction.
- Retained in one family but dropped in another: mixed-retention words were reviewed to determine whether the retained family context was still meaningful.
- Geography/place/customer-specific terms: most place-like terms were removed to avoid learning customer or location identity instead of project behavior; a small number of physical geography terms were retained when they described meaningful work context.
- Abbreviations/codes/unit-like terms: domain-specific construction abbreviations were retained when meaningful; workflow artifacts, stationing tokens, and non-specific abbreviations were removed.

## Final Approved Feature Set

Approved retained keyword family groups by family:

- Project keyword groups: 601 groups, 601 columns
- Contract keyword groups: 477 groups, 954 columns
- Item keyword groups: 862 groups, 2,586 columns

Total approved keyword family groups retained: 1,940

Total approved keyword feature columns retained: 4,141

Total final dataset columns: 4,216

## Important Interpretation Notes

- The approval decision is at the unique keyword level, but feature retention remains family-specific. For example, if a keyword passed the statistical screen for item descriptions but not project descriptions, only the item-description feature columns are retained.
- All non-keyword fields remain unchanged from the original flat feature table. This dataset only narrows the keyword-derived columns.
- The review is intended to reduce noise, leakage risk from customer/place-specific language, and mechanically common words that are unlikely to generalize. It does not prove that the retained keywords are predictive; that remains a modeling and validation question.
- Customer name is not added or reintroduced by this filtering step.
