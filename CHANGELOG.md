# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

