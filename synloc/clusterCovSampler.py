"""Versioned, parameter-only Gaussian mixture sampling for clusterCov."""

import numpy as np
from pandas import DataFrame


class ClusterCovSampler:
    """Sample a validated mixture without source records or source bounds.

    Loading validates numerical structure, not privacy or release approval.
    """

    VERSION = 1
    _FIELDS = {"version", "column_names", "weights", "means", "covariances"}

    def __init__(self, means, covariances, weights, columns):
        arrays = [np.asarray(value) for value in (means, covariances, weights)]
        if any(value.dtype.kind not in "fiu" for value in arrays):
            raise ValueError("Mixture parameters must be real numeric arrays.")
        means, covariances, weights = [value.astype(float, copy=True) for value in arrays]
        if means.ndim != 2 or min(means.shape) == 0:
            raise ValueError("means must have shape (components, columns), both nonempty.")
        k, p = means.shape
        if covariances.shape != (k, p, p) or weights.shape != (k,):
            raise ValueError("Invalid covariance or weight shape.")
        if not all(np.isfinite(value).all() for value in (means, covariances, weights)):
            raise ValueError("Mixture parameters must be finite.")
        if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("weights must be positive and normalized to one.")
        names = np.asarray(columns)
        if names.shape != (p,) or names.dtype.kind != "U":
            raise ValueError("column_names must be a one-dimensional Unicode string array.")
        if any(not name.strip() for name in names) or len(set(names.tolist())) != p:
            raise ValueError("column_names must be nonempty and unique.")

        self._factors = []
        for covariance in covariances:
            scale = float(np.abs(covariance).max())
            tolerance = np.finfo(float).eps * p * scale * 10
            if not np.allclose(covariance, covariance.T, rtol=0, atol=tolerance):
                raise ValueError("covariances must be symmetric.")
            if np.any(np.diag(covariance) < 0):
                raise ValueError("covariances must be positive semidefinite.")
            eigenvalues = np.linalg.eigvalsh(covariance)
            if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -tolerance:
                raise ValueError("covariances must be finite and positive semidefinite.")
            varying = np.flatnonzero(np.diag(covariance) > 0)
            constant = np.flatnonzero(np.diag(covariance) == 0)
            if constant.size and np.any(covariance[constant] != 0):
                raise ValueError("Constant dimensions must have zero covariance rows and columns.")
            values, vectors = np.linalg.eigh(covariance[np.ix_(varying, varying)])
            factor = vectors * np.sqrt(np.maximum(values, 0))
            self._factors.append((varying, factor))

        self.means_ = means
        self.covariances_ = covariances
        self.weights_ = weights
        self.columns = tuple(names.tolist())
        for value in (self.means_, self.covariances_, self.weights_):
            value.setflags(write=False)

    @classmethod
    def load(cls, path):
        """Load only schema-version 1 NPZ arrays, with pickle disabled."""
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("Expected a versioned NPZ model archive.")
        with archive:
            if len(archive.files) != len(cls._FIELDS) or set(archive.files) != cls._FIELDS:
                raise ValueError("Export fields must match the parameter-only schema.")
            version = archive["version"]
            if version.shape != () or version.dtype.kind not in "iu" or int(version) != cls.VERSION:
                raise ValueError("Unsupported clusterCov export version.")
            return cls(archive["means"], archive["covariances"], archive["weights"], archive["column_names"])

    def _export(self, path):
        # A file handle prevents NumPy from silently appending an extension.
        with open(path, "wb") as destination:
            np.savez_compressed(
                destination,
                version=np.asarray(self.VERSION, dtype=np.int64),
                column_names=np.asarray(self.columns, dtype=str),
                weights=self.weights_,
                means=self.means_,
                covariances=self.covariances_,
            )

    def sample(self, n, random_state=None):
        """Draw n rows; an integer seed reproduces direct and loaded draws exactly.

        No source min/max clipping or normalization factors are used. Zero rows
        are allowed. ``random_state`` follows ``numpy.random.default_rng``.
        """
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a nonnegative integer.")
        rng = np.random.default_rng(random_state)
        components = rng.choice(len(self.weights_), size=n, p=self.weights_)
        draws = self.means_[components].copy()
        for component, (varying, factor) in enumerate(self._factors):
            rows = np.flatnonzero(components == component)
            if rows.size and varying.size:
                noise = rng.standard_normal((rows.size, varying.size)) @ factor.T
                draws[np.ix_(rows, varying)] += noise
        return DataFrame(draws, columns=self.columns)
