"""Versioned, parameter-only Gaussian mixture sampling."""

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
        means, covariances, weights = [value.astype(float, copy=False) for value in arrays]
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
            diagonal = np.diag(covariance)
            if np.any(diagonal < 0):
                raise ValueError("covariances must be positive semidefinite.")
            varying = np.flatnonzero(diagonal > 0)
            constant = np.flatnonzero(diagonal == 0)
            if constant.size and (np.any(covariance[constant, :] != 0)
                                  or np.any(covariance[:, constant] != 0)):
                raise ValueError("Constant dimensions must have zero covariance rows and columns.")
            # Validate in correlation units so a large variance cannot hide an
            # indefinite or asymmetric block in smaller-unit dimensions.
            scales = np.sqrt(diagonal[varying])
            with np.errstate(over="ignore", invalid="ignore"):
                correlation = covariance[np.ix_(varying, varying)] / scales[:, None] / scales[None, :]
            tolerance = np.finfo(float).eps * p * 10
            if not np.isfinite(correlation).all():
                raise ValueError("covariances must be finite and positive semidefinite.")
            if not np.allclose(correlation, correlation.T, rtol=0, atol=tolerance):
                raise ValueError("covariances must be symmetric.")
            values, vectors = np.linalg.eigh(correlation)
            if not np.isfinite(values).all() or np.any(values < -tolerance):
                raise ValueError("covariances must be finite and positive semidefinite.")
            factor = (vectors * np.sqrt(np.maximum(values, 0))) * scales[:, None]
            self._factors.append((varying, factor))

        # Immutable backing prevents callers from re-enabling NumPy writes and
        # changing exported parameters without updating the cached factors.
        self._means = np.frombuffer(means.tobytes(), dtype=float).reshape(means.shape)
        self._covariances = np.frombuffer(covariances.tobytes(), dtype=float).reshape(covariances.shape)
        self._weights = np.frombuffer(weights.tobytes(), dtype=float).reshape(weights.shape)
        self._columns = tuple(names.tolist())

    @property
    def means_(self):
        """Read-only component means."""
        return self._means.view()

    @property
    def covariances_(self):
        """Read-only component covariances."""
        return self._covariances.view()

    @property
    def weights_(self):
        """Read-only component probabilities."""
        return self._weights.view()

    @property
    def columns(self):
        """Immutable output column names."""
        return self._columns

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
                raise ValueError("Unsupported mixture export version.")
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
