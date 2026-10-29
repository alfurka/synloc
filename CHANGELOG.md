# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
