# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-10-29
### Added
- Added `clusterCov.fit_model()`, parameter-only sampling, versioned NPZ export, and independent `ClusterCovSampler` loading.
- Added internal contributor, nonnegative-magnitude dominance, rank and constant-column diagnostics. Reports explicitly do not approve release; research exports require acknowledgement of unresolved review.
- Added a public SIPP 2018 four-quarter earnings/hours preparation and experiment workflow using only `clusterCov`, plus export and panel-regression behavioral tests.
- Recorded the first full public-panel experiment: 37,552 people, 12 configurations and 120 synthetic panels, with aggregate tables/figures and reproducibility manifests.

### Changed
- Shared clustering preparation between legacy `fit()` and model fitting; existing synthetic generation and clipping remain available.
- Exported means and covariances use original units even when clustering uses variance scaling. External sampling never depends on confidential extrema or preprocessing objects.

### Limitations
- Public-data research exports are not confidential-data clearance. Covariance disclosure, household dependence, count leakage through weights, and other full-model risks remain subject to review.
- The empirical example is an unweighted balanced analytical sample. Gaussian sampling is not constrained to preserve zero masses or valid hours; those shortcomings are measured rather than concealed.
- The first experiment preserves linear moments/regression coefficients closely but produces roughly 12–12.5% negative hours and loses nearly all exact zero hours; the unconstrained Gaussian model is not yet adequate for realistic panels containing nonworkers.

## [1.0.0] - 2026-05-02
### Added
- Added numeric-data validation with clear errors for non-numeric columns, duplicate columns, infinite values, and all-missing columns.
- Added utility metrics via `compareStats`, `kolmogorov_distances`, and `quality_report`.
- Added `compareStats()` and `qualityReport()` methods to fitted resamplers.
- Added pytest unit tests for resamplers, validation, metrics, and helper edge cases.
- Added Sphinx documentation source for GitHub Pages builds.

### Changed
- Updated package metadata for a stable 1.0 release.
- Made nearest-neighbor distance computation robust for small samples and large `K`.
- Made cluster sample-size allocation non-negative and robust for small requested sample sizes.
- Regularized covariance estimation for small, singular, or constant local samples.
- Preserved locally constant variables exactly when sampling from k-neighbor subsamples or clusters.

## [0.2.3] 

- Corrected an error in k-distance computation for `clusterResampling` class

## [0.2.1] - 2025-07-06
### Added
- Added k-distance computation for both kNNResampler and clusterResampler. After synthetic sample generation, both classes now compute and store the sum of distances to the K nearest neighbors for each observation in the original and synthetic data (using normalized data if normalization is enabled). Results are available as `self.data_distances` and `self.synthetic_distances`.

### Changed
- Enhanced kNNResampler parallel processing implementation for better performance
- Improved handling of neighbor indices in kNNResampler
- Added better error handling for synthetic data generation
- Added warning messages for non-uniform array results
