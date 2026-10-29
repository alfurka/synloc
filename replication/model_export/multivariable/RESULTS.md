# Multivariable SIPP panel experiment — 29 October 2026

## Bottom line

**The self-contained mixed-type clusterCov export works, but this first model is not a credible synthetic labour panel.** Its category decoding produces valid codes and bounds, and some means and serial correlations are close. However, it badly distorts zero masses, negative earnings, employment transitions, several multivariable coefficients, and all six predeclared key fixed-effects standard-error comparisons.

The initial configuration **failed the predeclared utility criteria** in [CRITERIA.md](CRITERIA.md). In accordance with those criteria, the experiment stopped after this configuration: no broader grid, unrestricted baseline model, alternative synthesis method, or selectively tuned decoder was run. Do not interpret valid category codes as evidence of realistic joint or longitudinal distributions. Neither this research output nor contributor checks establish privacy or regulatory approval.

## Source and panel

The input is the public 2018 SIPP panel, wave 1, which refers to January–December 2017. The supplied gzip matches the previously checked Census schema; SHA-256 is `d81a8b5debfe33ad371798cc7607cc43054bfa4efbd79e32bddd48ca3c801943`. This identifies the local copy, not an official authenticated release hash. Dictionary sources, universes, category codes, zero-filling and limitations are in [DATA_PREPARATION.md](DATA_PREPARATION.md).

| Selection step | People retained | Excluded at step |
|---|---:|---:|
| Input | 63,915 | — |
| Twelve distinct months | 63,283 | 632 |
| In survey frame all months | 62,768 | 515 |
| Monthly age 18–64 throughout | 36,790 | 25,978 |
| Required earnings available/valid | 36,657 | 133 |
| Required hours available/valid | **36,638** | 19 |
| Required demographics, other quarterly features valid | **36,638** | 0 additional |

The ordered, full selection flow, missingness, demographic profiles of excluded and retained people, and source provenance are in [preparation.json](summary/preparation.json). Selection before earnings/hours includes checks on constant valid sex/race/origin and December education; those steps excluded no additional people. The monthly age field is `TAGE_EHC`, not interview-time `TAGE`. A total of 119,536 retained person-months received a documented zero for each of earnings and hours only when **both** job and employment-status recodes confirmed no job. No employment or positive-income restriction was imposed. Quarterly earnings retain business losses.

The **14 distinct substantive variables** are Census-recorded sex, January age, year-end education, race and Hispanic origin, plus four-quarter earnings, hours, employment status, job count, insurance, marital status, household size, number of household members under 18, and the family income-to-poverty ratio. These are **41 wide columns**: five stable characteristics and nine varying measures × four quarters. One-hot encoding produces **113 Gaussian input dimensions**. `ESEX` is Census-recorded sex, not a measure of gender identity. Household measures are person-attached; the model does not synthesize coherent households. The sample is unweighted, balanced and complete-case, not a national population estimate.

## Model and verification

- Requested **50** K-means clusters with a **100-person minimum**; **42** components remain after the existing reassignment heuristic. Component sizes range from **105 to 3,483** people. Ten synthetic datasets each contain 36,638 people and four quarters.
- The mixed schema contains fixed public codes, **descriptive labels**, public bounds, and the explicit child-count ≤ household-size constraint. One-hot categories are decoded by deterministic argmax; integers are rounded and bounded; hours and other bounded variables use public dictionary bounds. Earnings are not floored because negative business losses can be real. Employment status refers to the quarter's last month; it is not used to erase earnings or hours for the whole quarter.
- Clustering scales each encoded feature for distance, but Gaussian means and covariances are estimated in encoded original units. No confidential source standard deviations, raw extrema, source rows, IDs, labels assigning people to clusters, or fitted preprocessors are needed in the export. Public code descriptions are included to make the categorical values interpretable.
- Direct and loaded samples are exactly equal at the same seed; a fresh process loading only the research artifact produces the same decoded sample. The export is **1,429,383 bytes**. The parameter format remains distinct from the prior numeric-only format and does not use pickle.
- Every component's empirical centered-data matrix is rank deficient (ranks 41–78 of 113); full one-hot blocks have structural sum-to-one dependencies and some categories are constant. This rank observation alone is neither an error nor a privacy assessment. The existing covariance regularization operates in mixed original units, so its effects on scaled/dummy features warrant scrutiny.
- **103 tests passed** and `pip check` passed. Tests include fixed-effects coefficient recovery, categorical reconstruction, tie-breaking, bounds, structural constraints, schema validation, independent loading and SIPP preparation.

## Distributional and validity results

Values below are original person-quarter proportions versus averages over ten synthetic panels. Synthetic SD is conditional on this one fitted model; it does not include uncertainty in fitting the model.

| Outcome | Original | Synthetic average |
|---|---:|---:|
| Exactly zero quarterly earnings | **27.35%** | **0.00%** |
| Negative quarterly earnings | **0.78%** | **21.33%** |
| Exactly zero weekly hours | **26.52%** | **11.62%** |
| Negative decoded weekly hours | 0.00% | 0.00% |
| Zero jobs but quarter-end employment code indicating a job | 0.00% | **3.05%** |
| Positive jobs but quarter-end code indicating no job | 0.09% | **0.58%** |

The absence of negative **decoded** hours results from clipping to the public range: **11.62% of raw person-quarter hour draws were clipped**. This repair makes values valid but does not reproduce the original zero mass. No artificial relationship was imposed between a quarter's income/hours and a last-month employment indicator. The two job/status recodes can disagree in the source, so rates are measured rather than assumed automatically impossible.

Sex and broad insurance prevalence are much closer: Census sex code 2 is 51.56% original versus 51.42% synthetic; insured status is 85.35% versus 85.47%. But this does not demonstrate correct joint categories or dynamics. An average of **85.4% of raw sex one-hot score blocks** and **94.9% of education blocks** were not valid one-hot vectors; argmax decoding repairs the representation. That repair is part of the generator, not evidence that the latent scores represent genuine categories.

Among 35 predeclared category/zero prevalence checks, **2 failed**, both the earnings and hours zero rates. Eleven of 96 eligible employment, insurance and marital transition checks failed. For employment code 3 (job all month, unpaid absence due to layoff), the original conditional probability of remaining in code 3 was **32.49%**, while the synthetic mean was **0% in the eight replicates with any such starting observations**; the other two had no such starting observations and count as failures rather than passes. Rare categories and small cell denominators warrant cautious interpretation.

Averages preserve some simpler features: quarterly earnings mean is $10,645 original versus $10,635 synthetic, and the adjacent-quarter earnings correlation is 0.913 versus 0.912. These are not independent validation of a Gaussian mixture fitted to moments. Original mean quarterly earnings for Census sex code 1 is $13,594 versus $13,187 synthetic; for code 2, $7,874 versus $8,223. Full subgroup and joint-category comparisons and transition matrices are in [utility_comparison.csv](summary/utility_comparison.csv). Spell counts are *observed runs* within four quarters and can be censored at both ends, not complete underlying spell durations.

## Multivariable regression results

Both original and all ten synthetic samples identify the specified pooled and individual-fixed-effects models; no terms were silently dropped. Outcomes are quarterly earnings. Regressors include hours, job count, household size, number of household members under 18, family poverty ratio, employment, insurance and marital categories, and quarter effects, plus a sex-by-hours interaction. The pooled model also includes Census sex, baseline age, race, origin, and education. For regression contrasts only, education is grouped a priori as less than high school, high school, some college/associate's, and bachelor's or higher; the synthesis retains all 16 source codes. Fixed-effects models absorb stable person-characteristic main effects.

| Individual fixed-effects term | Original coefficient | Original person-clustered CR0 SE | Synthetic mean coefficient | Mean synthetic CR0 SE |
|---|---:|---:|---:|---:|
| Hours (sex code 1 reference) | 279.99 | 21.82 | 263.57 | 6.92 |
| Number of jobs | 322.42 | 378.96 | 21.44 | 105.08 |
| Household size | −397.14 | 81.49 | −74.91 | 51.95 |
| Household members under 18 | 608.66 | 122.54 | 120.30 | 76.88 |
| Family poverty ratio | 939.75 | 97.81 | 1,016.90 | 30.72 |
| Sex code 2 × hours | −36.58 | 23.25 | −18.57 | 8.94 |

**All six key FE SE ratios** fell outside the predeclared 0.75–1.25 range; synthetic-to-original ratios were roughly 0.28–0.64. Across both full earnings regressions, **25 of 54** original coefficients exceeded the predeclared tolerance of two original person-clustered SEs (or could not be compared); no regression became unidentified. The complete [regression comparison](summary/regression_comparison.csv) includes all coefficients, Monte Carlo SDs and conditional SEs.

The original-panel FE hours SE is **21.79** under a sensitivity analysis clustering people by their Q1 household, compared with **21.82** clustering by person. Q1 household clustering is an approximation because households can change. There is no corresponding household-clustered synthetic estimate: generated people do not have coherent synthetic households. [Full original sensitivity](summary/original_household_se_sensitivity.csv).

For a further nonlinear-outcome check, the same pooled covariates were used in a descriptive **linear probability model for positive quarterly earnings**. Its hours coefficient was **0.00184** originally versus **0.00559** on average synthetically. [All original and replicate estimates](summary/positive_earnings_regressions.csv) are available. This is not a causal model, and because the benchmark variables were included in synthesis it is not an out-of-sample prediction test.

All regressions are descriptive, not causal. Person/household-clustered CR0 SEs have no finite-sample adjustment; synthetic CR0 SEs do **not** automatically account for model-estimation or synthesis uncertainty. Replicate SD reflects draws conditional on one fitted model. Education measured at year-end may not have been known at the beginning of the year.

## Disclosure assessment

Final contributor-count diagnostics pass the configured minimum of 100 rows, assuming one person per row. In the internal encoded-feature audit, 3,033 nonnegative component-feature dominance diagnostics passed, **456 failed**, 1,129 zero-total checks were unassessed, and 128 signed-feature checks were not applicable. These are not counts of cleared covariance elements or a complete policy assessment. Exact component sizes, means, covariances and weights can reveal information; rare-category homogeneity, household dependence, combined release and reconstruction/inference remain unresolved. No formal privacy guarantee, ABS clearance or confidential-data release approval is claimed. The research acknowledgement `reviewed=True` is not an approval.

## Reproduction and evidence

[README.md](README.md) provides commands. The saved [manifest](summary/manifest.json), [diagnostic gates](summary/diagnostic_gates.json), [model summary](summary/model_summary.json), aggregate CSVs and selection metadata describe the executed run. The internal fitted model, raw records, original IDs, person-level outputs and full audit remain locally ignored. Aggregates here are based on a public-use input and would themselves require review in a confidential-data context.

**Research decision:** this is a clear negative first experiment, not merely a minor formatting defect. Addressing realistic zero-heavy and categorical trajectories within the clustering-only constraint is an open methodological question; clipping or reporting only favorable linear moments would not establish a solution.
