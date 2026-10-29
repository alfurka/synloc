# 2018 SIPP multivariable panel preparation

## Source and reproducibility

The input is the user-supplied, pipe-delimited **2018 SIPP panel, wave 1**, which covers January–December **2017**, as specified by the [Census release](https://www.census.gov/programs-surveys/sipp/data/datasets/2018-data/2018.html). The local gzip file's previously recorded SHA-256 is `d81a8b5debfe33ad371798cc7607cc43054bfa4efbd79e32bddd48ca3c801943` (164,090,268 bytes). This hash identifies the local file, not an authenticated official release hash; the release page advertises a later revision. Each preparation recomputes the input checksum and validates every row's panel, wave and month. The prior baseline compared the full 4,942-column header and order with the [official schema](https://www2.census.gov/programs-surveys/sipp/data/datasets/2018/pu2018_schema.json).

Field definitions below are from the official [2018 combined dictionary v3](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2018/2018_SIPP_Data_Dictionary_v3.pdf), linked at the [dictionary index](https://www.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/data-dictionaries-2018.html). The dictionary states numeric not-in-universe is `.`; this local file also uses empty fields. Both are missing, not zeros. Census allocation flags can indicate imputation and source earnings may be topcoded; this preparation neither undoes Census processing nor filters allocation flags. The [TPEARN user note](https://www.census.gov/programs-surveys/sipp/tech-documentation/user-notes/2018-usernotes/2018-TPEARN-hist-comp.html) distinguishes month-length-varying `TPEARN` from the alternative. No weights, inflation adjustment or re-imputation are applied.

From the repository root, with dependencies installed:

```sh
.venv/bin/python replication/model_export/multivariable/prepare_sipp.py \
  --input "$SIPP_INPUT" \
  --output replication/model_export/multivariable/prepared
```

`wide.csv` is one person per row with the five stable columns followed by four quarters each of the nine measures listed below; it contains no identifiers. `long.csv` contains four rows per person with `person_id` and the quarter-end `household_id` (`SSUID:SHHADID`) for **internal** clustered analyses. A household can change across quarters; Q1 grouping is a sensitivity approximation, not a lifelong household identifier. Long and wide are sorted consistently by original person keys. `metadata.json` contains the checksum, fixed public category labels, missingness, selection flow and subgroup impacts. All three outputs are internal and ignored; do not publish them or exact diagnostics without review. The committed `schema.json` is a public fixed model contract, not estimated from the input. It carries decoder support bounds; earnings decoder bounds are deliberately null, so no data-derived clipping is introduced.

## Fourteen measures and dictionary universes

All monthly categories and counts are taken at the **last calendar month of each quarter** (March, June, September, December), unless otherwise noted. The complete-case checks apply to these four month-ends rather than needlessly requiring every intervening status. The person-level age and month-summed or month-averaged variables do require all 12 months.

| Wide base | Source | Definition and treatment |
|---|---|---|
| `sex` | `ESEX` | Census sex, all persons; use January value only after verifying valid and constant in all months. It is not gender identity. |
| `age` | `TAGE_EHC` | Monthly age during reference year; require integer 18–64 throughout and use January as the fixed baseline. `TAGE` is age as of last birthday at interview, not the monthly age history. No birthdays are fabricated. |
| `education` | `EEDUC` | Highest education completed **by December of the reference year**, universe age 15+; take December. It need not have been known in Q1. |
| `race` | `ERACE` | Mutually exclusive Census recode, all persons; require valid and constant all months, use January. Residual is not a single race. |
| `origin` | `EORIGIN` | Hispanic origin, all persons; require valid and constant all months, use January, separately from race. |
| `earnings` | `TPEARN` | Sum of three monthly earnings and business profits/losses, nominal dollars; at least one job on lines 1–8 in the source universe. Negative business losses are valid and retained. Each observed monthly value must lie in the dictionary range −99,999,999 to 99,999,999. |
| `hours` | `TMWKHRS` | Equal-month arithmetic mean of three weekly-hour rates (0–99.99), not quarterly hours; source universe at least one job on lines 1–7. Do not assume the job-line universes are identical. |
| `employment` | `RMESR` | Monthly employment status, universe respondents age 15+; end-of-quarter status, not a status for the whole quarter. |
| `jobs` | `RMNUMJOBS` | Number of jobs held in month, respondents age 15+, integer 0–9. |
| `insurance` | `RHLTHMTH` | Insured by private or public coverage during month, all persons. |
| `marital` | `EMS_EHC` | Marital status **at this month**, universe monthly age 15+. `EMS` instead describes status at interview, with universe present at interview; it is not substituted as a monthly history. |
| `household_size` | `RHNUMPER` | Persons in household this month, all persons, 1–20. Person-attached household characteristic, not an independently reconstructed household. |
| `children` | `RHNUMU18` | Persons under 18 in household this month, all persons, 0–20 and no greater than household size; not necessarily the respondent's own children. |
| `poverty_ratio` | `TFINCPOV` | Family income-to-poverty ratio for the month **excluding Type 2 persons**, universe `RFPOV > 0`. Use the source ratio without dividing by 100; accept documented 0–999.999 and require a valid threshold `RFPOV` in 1–7,000. Missing/out-of-universe is not zero. The separate `TFINCPOVT2` includes Type 2 persons and is not used. Household-count and family-poverty units are different; do not infer that they count the same people. |

Type 2 people are separately represented in the [Type 2 dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2018/2018_SIPP_Type_2_Data_Dictionary.pdf). The original respondents' `RHNUMPER`/`RHNUMU18` are used as published, with no attempted Type 2 reconstruction. Household variables are attached to individuals and this does not generate coherent households. In the combined dictionary, the `TFINCPOV` range describes valid numeric values; no separate numeric missing or special code is specified. A zero is an observed ratio, not missing; its causes, including nonpositive family income where applicable, are not identified by the ratio alone.

## Public categories

All code sets below are fixed from the dictionary, including categories absent or rare in a particular sample. `schema.json` contains codes; `metadata.json` contains these source labels. Census codes remain integers, not ordinal distances.

* `sex`: 1 Male; 2 Female.
* `education`: 31 less than first grade; 32 first–fourth grade; 33 fifth–sixth; 34 seventh–eighth; 35 ninth; 36 tenth; 37 eleventh; 38 twelfth without diploma; 39 high school graduate or equivalent; 40 some college less than one year; 41 one or more years college without degree; 42 associate's; 43 bachelor's; 44 master's; 45 professional; 46 doctorate.
* `race`: 1 White alone; 2 Black alone; 3 Asian alone; 4 residual.
* `origin`: 1 Hispanic, Spanish or Latino; 2 not Hispanic, Spanish or Latino.
* `employment`: 1 job entire month and worked all weeks; 2 job all month with unpaid absence not due to layoff; 3 job all month with unpaid absence due to layoff; 4 job some weeks without layoff or job search; 5 job some weeks with layoff or job search; 6 no job, layoff or job search all weeks; 7 no job, layoff or job search some but not all weeks; 8 no job or layoff/job search.
* `insurance`: 1 insured; 2 not insured.
* `marital`: 1 married, spouse present; 2 married, spouse absent; 3 widowed; 4 divorced; 5 separated; 6 never married.

## Selection, missingness, and zero decisions

Person-month duplicates, malformed keys, invalid month or wrong panel/wave fail the run. The ordered flow starts with all persons, then requires 12 unique months, all-month survey-frame membership (`RIN_UNIV=1`), age 18–64 each month, valid and constant sex/race/origin, valid December education, and each required earnings, hours, employment, jobs, insurance, marital, household-size, children and poverty value. Counts of persons and person-months removed at **every** step, retained and excluded counts by sex, January/first-observed-month age group, race, origin and December education, and per-variable missing/invalid counts are recorded in `metadata.json`. For partial-year people lacking December, education is explicitly grouped as missing. Pre-selection variable diagnostics overlap and are not additive; sequential exclusions are additive. `RFPOV` inapplicability and an invalid children/household relationship are part of the relevant invalid count. No imputations are made for missing covariates.

Only if **both** `RMNUMJOBS=0` and `RMESR` reports no job all month (codes 6–8) does the script fill a missing `TPEARN` or `TMWKHRS` with zero. Observed values, including losses, are never overwritten. The raw file inspection found 2,646 person-months with zero jobs but `RMESR=1`, illustrating why the agreement rule matters; missing labor outcomes in those disagreements remain missing and cause exclusion if required. Filled counts for all input and retained months, disagreement counts, and retained negative earnings months are reported. Other cases of missing values in a job universe remain missing, rather than becoming zero. No requirement of employment or positive earnings is imposed. A quarter-end no-job status does **not** imply zero quarterly earnings or zero quarter-average hours: someone may have worked earlier in that quarter. No such cross-period constraint is imposed.

A full source-column scan (not a full preparation run) found 763,186 person-months, exclusively panel 2018 wave 1; observed category codes were within the published sets, and sex, race, origin and education were constant within each recorded person. There were 1,446 missing `TFINCPOV` values, all with missing `RFPOV`, 30,387 observed zero ratios, and no negative or above-dictionary-maximum ratios. Of the zero ratios, 3,575 coincided with negative `TFTOTINC`. A source-wide comparison with `TFTOTINC / RFPOV` showed median absolute difference approximately 0.000483, rather than a 100-times scaling; there were no observed values at 999.999. This check describes the supplied public file, not a substitute for Census's definitions. The scan included all ages, not just the eligible population. The preparation's own flow, required-month missingness and subgroup tables are the authoritative numbers for any run; these source-wide numbers do not imply selected-sample representativeness.

This is an unweighted, balanced, complete-case analytical sample. It excludes partial-year/out-of-frame people, people crossing the age boundaries, and those lacking a required value or with inconsistent stable characteristics. Selection can alter demographic and employment composition; compare the recorded retained and excluded profiles before interpreting the model. Public data, fitted parameters and internal person-level files are not interchangeable disclosure approvals.
