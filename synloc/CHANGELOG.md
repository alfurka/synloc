# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2]
### Changed
- Enhanced kNNResampler parallel processing implementation for better performance
- Improved handling of neighbor indices in kNNResampler
- Added better error handling for synthetic data generation
- Added warning messages for non-uniform array results
- Removed `synthia_examples.py` and the dependency on the `synthia` package. 

## [0.1.2] - 2025-05-06
### Added
- Initial public release
- Implemented kNNResampler class for synthetic data generation
- Added LocalCov and LocalGaussianCopula methods
- Included parallel processing support using joblib
- Added data visualization tools with comparePlots method

[Unreleased]: https://github.com/alfurka/synloc/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/alfurka/synloc/releases/tag/v0.1.2