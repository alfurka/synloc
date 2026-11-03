# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-03
### Added
- Added optional parameter-only model fitting, export and independent sampling for `clusterCov` and `LocalCov`; `LocalCov.prepare_export()` can select all components or a reproducible subset while finding neighbours in the full input.
- Added versioned NPZ loading with pickle disabled and read-only fitted parameters.
- Added internal contributor-count and disclosure diagnostics. Export requires an explicit review acknowledgement; neither the diagnostics nor successful export establish privacy or release approval.
- Added model-export regression coverage and a focused export/loading guide with compiled HTML for GitHub Pages.

### Fixed
- Made explicit kNN seeds control anchor selection and NumPy/SciPy callback draws consistently across worker counts while restoring the caller's NumPy RNG state.
- Accepted NumPy integer counts, rejected boolean/fractional counts and complex inputs, and cleared stale outputs after failed refits.
- Preserved locally constant decimal values exactly and validated/factored covariances in variance-scaled units.

### Compatibility
- Preserved the basic constructors and `fit()` workflows; `LocalCov` has a trailing optional `random_state`.
- Exported means and covariances remain in original units; exported sampling does not clip to confidential source bounds.
- Exact seeded samples can differ from earlier numerical implementations; direct and reloaded samples agree within the same build and numerical environment.
- Existing published-paper replication materials are unchanged.

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
