# SIPP source and preprocessing

## Verified source

The input is the user-supplied Census **2018 SIPP, panel 2018 wave 1** primary public-use file, `pu2018.csv.gz`. Despite its name, it covers **January–December 2017**, not 2018. The [official release page](https://www.census.gov/programs-surveys/sipp/data/datasets/2018-data/2018.html) explicitly identifies that reference period and panel/wave. It advertises a July 24, 2025 Social Security processing update; we do not assume that the local file is that revision merely from its filename. Each run records its SHA-256, byte length, and observed panel/wave values. There is no reference-calendar-year field in the official schema; year 2017 comes from the release documentation conditional on the validated panel/wave.

On 2026-10-29, the local header's **4,942 names and their order** matched the [official schema](https://www2.census.gov/programs-surveys/sipp/data/datasets/2018/pu2018_schema.json). Small-sample research inspected the first 240 records, not the complete analytical sample. The schema hash and compact verified dictionary excerpts are saved in `sipp_source_metadata.json`; these describe source research, not an executed preparation or empirical result. The script does not access the network. Runtime checks apply to every input row.

The file is **pipe-delimited**, despite the `.csv` extension. The script reads only ten selected columns in chunks (default 25,000 rows), then combines the narrow table in memory. It does not load the thousands of unused fields, duplicate the raw file, or require packages beyond pandas and NumPy.

## Variables and universes

Definitions are from the [ID dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2018/2018_SIPP_ID_Variables_Data_Dictionary.pdf), [demographics dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2018/2018_SIPP_Demographics_Data_Dictionary.pdf), and [labor-force dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2018/2018_SIPP_Labor_Force_Data_Dictionary.pdf), linked by the [2018 dictionary index](https://www.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/data-dictionaries-2018.html).

| Column | Definition and use |
| --- | --- |
| `SSUID`, `PNUM` | Sample-unit ID and person number, jointly identifying a person within this panel. Preserve ID strings and leading zeros; neither is a model feature. |
| `SPANEL`, `SWAVE` | Require 2018 and 1 on every row; never infer these solely from the filename. |
| `MONTHCODE` | Reference month, integer 1–12. Person-month duplicates are fatal, not silently dropped. |
| `RIN_UNIV` | Monthly survey-frame membership: 1 in universe, 2 not in universe. Require 1 throughout. |
| `TAGE` | Age as of last birthday. Require 18–64 in all 12 months. |
| `RMNUMJOBS` | Number of jobs within the reference month, range 0–9, universe age 15+. Zero documents a no-job month. |
| `TPEARN` | Sum of earnings **and business profits/losses** from all jobs, varying with days in the month. Universe: at least one job on job-lines 1–8 during the month. Negative values can be genuine losses. |
| `TMWKHRS` | Average **hours per week** at all jobs held during the reference month, range 0–99.99. Universe: at least one job on job-lines 1–7 during the month. This is not monthly total hours. |

The labor-force PDF entries are on PDF pages 1366 (`TMWKHRS`), 1367 (`RMNUMJOBS`), and 1375 (`TPEARN`). The different earnings/hours job-line universes are not assumed interchangeable: missing hours for someone with a job remain missing and trigger complete-case exclusion.

Census [explains the earnings alternatives](https://www.census.gov/programs-surveys/sipp/tech-documentation/user-notes/2018-usernotes/2018-TPEARN-hist-comp.html): `TPEARN` varies with month length; `TPEARN_ALT` smooths that variation. We deliberately use `TPEARN`. These public values may be imputed or topcoded; no attempt is made to reverse Census processing, and allocation flags are not used to restrict the sample.

## Selection and missingness

Apply this ordered person-level flow, recorded with both retained and excluded person/person-month counts in `metadata.json`:

1. Start with every input person. Reject malformed identifiers, invalid months, other panels/waves, or duplicate person-month keys as input errors.
2. Retain exactly 12 distinct months. With valid unique month codes, this is January through December.
3. Require survey-frame membership in every month.
4. Require age 18–64 in every month. This is the implemented working-age definition rather than the provisional age band discussed during planning.
5. Require finite monthly earnings and weekly-hours rates after the narrowly defined zero fill below, with hours within their documented 0–99.99 range. Drop a person if any month fails; never silently aggregate an incomplete quarter.

Dictionary numeric not-in-universe values use `.`. The supplied file's inspected no-job adult records instead have blank `TPEARN`/`TMWKHRS`; both blanks and dots are parsed as missing. **Only when `RMNUMJOBS=0`** is a missing earnings or hours value replaced by zero. An observed value is never overwritten, including an observed negative earnings value. Other missing outcomes remain missing, not zeros. Both input and retained structural-zero counts are reported.

There is **no positive-earnings or employment requirement**: people with intermittent employment and year-round nonworkers remain eligible. There is **no nonnegative earnings filter**: negative business earnings remain in the sample and their counts are reported. No log transformation, clipping, winsorization, scaling, inflation adjustment, or outlier trimming is applied. Hours outside their documented nonnegative range are invalid outcomes, not values to clip.

## Quarterly definitions and files

For each retained person and calendar quarter:

- `earnings_q1`–`earnings_q4` are the sum of three monthly `TPEARN` amounts, in nominal 2017 dollars.
- `hours_q1`–`hours_q4` are the **equal-month arithmetic mean** of the three `TMWKHRS` rates, in hours/week. This explicitly measures mean monthly weekly-hours intensity, not actual quarterly hours or a calendar-day-weighted quarterly weekly rate. Summing these monthly rates would give the wrong units; reconstructing total quarterly hours requires more granular timing information.

Every retained quarter must have three months, and every person four quarters. Output rows are deterministic, sorted by `SSUID`, then `PNUM`.

- `wide.csv`: exactly the eight numeric feature columns, earnings first then hours, quarter order 1–4; no index or person ID.
- `long.csv`: `person_id,quarter,y,h`, four rows per person. Internal `person_id` joins `SSUID:PNUM`; `y` and `h` use the same quarterly definitions. This is an internal diagnostic table, not a shareable model artifact.
- `metadata.json`: source checksum and provenance, software versions, ordered sample flow, feature definitions, structural-zero/negative-earnings counts, dimensions, and limitations. No claim of an empirical sample size is made before execution.

Prepared data, IDs, and experiment outputs are ignored. Keep custom output directories ignored too. Only a separately reviewed parameter-only model is an export candidate; the input source's public-use clearance does not automatically approve a fitted model, metadata, or diagnostics for disclosure.

## Interpretation and selection bias

This is an **unweighted complete-case public-use demonstration**, not a population estimate. Balanced-frame selection disproportionately omits partial-year and out-of-frame people; completeness restrictions may select on employment histories and reporting/imputation patterns. Requiring working age throughout excludes those turning 18 or 65 during 2017. Survey and replicate weights are deliberately not used, and the source may already contain imputed/topcoded values. All utility comparisons refer to this retained panel, not all US adults or an unreleased confidential SIPP file.
