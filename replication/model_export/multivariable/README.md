# Multivariable clusterCov SIPP experiment

**Executed result:** [RESULTS.md](RESULTS.md). The first model has 14 substantive measures, 41 wide columns, 113 encoded Gaussian dimensions, and 36,638 people. The model/export round trip works, but the decoded synthetic panel fails the predeclared utility criteria for zeros, transitions, regression coefficients and standard errors. These are public-data research results, not confidential-data clearance.

## Data and interpretation

[DATA_PREPARATION.md](DATA_PREPARATION.md) describes the Census sources, field definitions, codes, reference dates, sample selection, missingness, no-job zero filling, age variable, original-unit earnings/hours, household information and limitations. The input is public SIPP 2018 wave 1 (2017 reference year). The Census sex field records sex, not gender identity. The fixed [schema.json](schema.json) records public codes, category descriptions, and bounds independently of sample observations.

Five characteristics—sex, January age, December education, race and origin—are stable per person. Nine other measures are four-quarter trajectories. One-hot latent scores are decoded by argmax, counts are rounded/clipped to public bounds, and a child-count ≤ household-size constraint is applied; earnings remain signed. No model observes source person or household IDs. Applied repairs are counted rather than hidden. Household measures are attached to individuals; no coherent synthetic households are created.

[CRITERIA.md](CRITERIA.md) specifies the diagnostic thresholds declared before running the first configuration. It requires stopping rather than tuning or selectively narrowing the sample after failure. Rare groups, transition discrepancies, within-person inference and unresolved disclosure risks are central findings.

## Reproduce

From the package repository root, in an environment installed with `pip install -e . pytest`, point `SIPP_INPUT` to the public compressed pipe-delimited source file. Source and generated person-level files must remain outside version control.

```sh
python -m replication.model_export.multivariable.prepare_sipp \
  --input "$SIPP_INPUT" \
  --output replication/model_export/multivariable/prepared

python -m replication.model_export.multivariable.run_multivariable \
  --input replication/model_export/multivariable/prepared/wide.csv \
  --long replication/model_export/multivariable/prepared/long.csv \
  --schema replication/model_export/multivariable/schema.json \
  --output replication/model_export/multivariable/results

python -m replication.model_export.multivariable.summarize \
  replication/model_export/multivariable/results

python -m replication.model_export.multivariable.positive_earnings \
  --input replication/model_export/multivariable/prepared/wide.csv \
  --model replication/model_export/multivariable/results/mixture.npz \
  --output replication/model_export/multivariable/results

python -m pytest tests -q
```

Default run: 50 requested clusters, minimum 100 contributors, 10 independent synthetic samples, fixed fit seed and sampling seeds beginning 20261029, two numerical threads. The long original data are needed only for Q1 household-clustered SE sensitivity; they never enter the export. An empirical rerun may differ across Python/numerical-library versions; versions and input/schema checksums are recorded in the manifest.

`prepared/` and `results/` are ignored: they contain person-level data, sample IDs, raw scores, fitted exports and/or internal audit evidence. `summary/` contains the public-input aggregate evidence used for the report. In a confidential-data setting all diagnostics, derived metadata, and summaries require independent disclosure review, even if they do not contain raw rows. The synthetic panel, audit evidence and schema must not be presented as automatically approved for release.

## Analysis details

Both earnings regressions use quarterly earnings, weekly hours, job count, household size, children, poverty ratio, employment/insurance/marital dummies, quarter effects and Census-sex-by-hours interaction. The pooled specification additionally includes age, sex, education, race and origin. Regression-only education categories are < high school, high school, some college/associate's, and bachelor's or higher; the synthetic model retains the original 16 category codes. Stable main effects are absorbed in the person fixed-effects specification. Regression rank deficiencies are reported, never silently removed. Person-clustered CR0 standard errors do not account for fitting/synthesis uncertainty.

The additional positive-earnings linear probability model is a descriptive nonlinear-outcome check, not causal or out-of-sample validation. Original Q1 household-clustered SEs are a sensitivity check; synthetic household-clustered SEs are unavailable because the model does not generate households. `utility_comparison.csv` includes mean, variance, quantile, subgroup, joint-category, transition, spell and consistency metrics; observed spells are censored at the four-quarter boundaries.
