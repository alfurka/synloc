from .kNNResampler import kNNResampler
from .clusterResampler import clusterResampler
from .clusterCovSampler import ClusterCovSampler
from scipy.stats import multivariate_normal
from pandas import DataFrame
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .tools import fill_na_with_median

def _regularized_covariance(data):
    """Return a symmetric positive covariance matrix for sampling."""
    n_features = data.shape[1]
    if n_features == 0:
        raise ValueError("Cannot estimate a covariance matrix with zero columns.")

    if data.shape[0] <= 1:
        variances = data.var().fillna(0.0).to_numpy(dtype=float)
        variances = np.where(variances > 0, variances, 1e-12)
        return np.diag(variances)

    covariance = data.cov().fillna(0.0).to_numpy(dtype=float)
    covariance = np.atleast_2d(covariance)
    covariance = (covariance + covariance.T) / 2.0
    scale = max(float(np.trace(covariance)) / max(n_features, 1), 1.0)
    covariance = covariance + np.eye(n_features) * scale * 1e-10
    return covariance

def _local_constant_mask(data):
    """Return columns that are exactly constant in this local sample."""
    return data.max(axis=0).to_numpy(dtype=float) == data.min(axis=0).to_numpy(dtype=float)

def _component_parameters(data):
    """Fit one Gaussian and summarize its source rows for disclosure review."""
    columns = list(data.columns)
    constant = _local_constant_mask(data)
    varying = np.flatnonzero(~constant)
    mean = data.mean().to_numpy(dtype=float, copy=True)
    mean[constant] = data.iloc[0].to_numpy(dtype=float)[constant]
    covariance = np.zeros((len(columns), len(columns)))
    if varying.size:
        covariance[np.ix_(varying, varying)] = _regularized_covariance(data.iloc[:, varying])
    top_one, top_two = {}, {}
    for name in columns:
        values = np.sort(data[name].to_numpy(dtype=float))[::-1]
        total = values.sum()
        assessable = values[-1] >= 0 and total > 0
        top_one["top1_share_" + name] = values[0] / total if assessable else np.nan
        top_two["top2_share_" + name] = values[:2].sum() / total if assessable else np.nan
    diagnostics = {
        "count": len(data),
        "rank": int(np.linalg.matrix_rank(data.to_numpy(dtype=float) - mean)),
        "constant_columns": tuple(name for name, flag in zip(columns, constant) if flag),
        **top_one,
        **top_two,
    }
    return mean, covariance, diagnostics

def _diagnostics_frame(rows):
    """One row per component, in the same order as the fitted means."""
    return DataFrame(rows).rename_axis("component")

def _draw_local_normal(data, size=None):
    """Draw from local covariance while preserving locally constant columns."""
    means = data.mean(axis=0).to_numpy(dtype=float, copy=True)
    constant_mask = _local_constant_mask(data)
    means[constant_mask] = data.iloc[0].to_numpy(dtype=float)[constant_mask]
    varying_indices = np.flatnonzero(~constant_mask)

    if size is None:
        draw = means.copy()
        if varying_indices.size == 0:
            return draw

        varying_data = data.iloc[:, varying_indices]
        covariance = _regularized_covariance(varying_data)
        varying_mean = means[varying_indices]
        varying_draw = multivariate_normal.rvs(mean=varying_mean, cov=covariance)
        draw[varying_indices] = np.asarray(varying_draw, dtype=float).reshape(-1)
        return draw

    size = int(size)
    draws = np.tile(means, (size, 1))
    if varying_indices.size == 0:
        return draws

    varying_data = data.iloc[:, varying_indices]
    covariance = _regularized_covariance(varying_data)
    varying_mean = means[varying_indices]
    varying_draws = multivariate_normal.rvs(mean=varying_mean, cov=covariance, size=size)
    draws[:, varying_indices] = np.asarray(varying_draws, dtype=float).reshape(size, varying_indices.size)
    return draws

class _LocalCovExport:
    """A fixed set of selected local components and their diagnostics."""

    def __init__(self, model, diagnostics):
        self._model = model
        self._diagnostics = diagnostics

    @property
    def means_(self):
        return self._model.means_

    @property
    def covariances_(self):
        return self._model.covariances_

    @property
    def weights_(self):
        return self._model.weights_

    @property
    def columns(self):
        return self._model.columns

    def disclosure_diagnostics(self):
        """Return per-component source diagnostics; see :meth:`clusterCov.disclosure_diagnostics`."""
        return self._diagnostics.copy()

    def sample(self, n, random_state=None):
        """Sample the selected parameter mixture without source-bound clipping."""
        return self._model.sample(n, random_state=random_state)

    def export_model(self, path):
        """Write the selected parameters to a versioned NPZ file."""
        self._model._export(path)


class LocalCov(kNNResampler):
    """This is a `method` for `clusterResampler` class to create synthetic samples from the multivariate normal distribution with the estimated covariance matrix.

    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param K: The number of the nearest neighbors used to create synthetic samples, defaults to 30
    :type K: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    :param n_jobs: The number of jobs to run in parallel, defaults to -1
    :type n_jobs: int, optional
    :param Args_NearestNeighbors: `NearestNeighbors` function arguments can be specified if needed. See scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html , defaults to {}
    :type Args_NearestNeighbors: dict, optional
    :param random_state: Optional integer seed for anchor selection and NumPy-based draws.
    :type random_state: int, optional
    """    
    @staticmethod
    def method(subsample:DataFrame):
        """Estimates covariance matrix and draw samples from the estimated multivariate normal distribution.

        :param subsample: A subsample defined by the `kNNResampler` class.
        :type subsample: pandas.DataFrame
        :return: Synthetic values. 
        :rtype: numpy.darray
        """        
        return _draw_local_normal(subsample)

    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, n_jobs:int=-1, Args_NearestNeighbors:dict={}, random_state=None):
        super().__init__(data, self.method, K, normalize, clipping, n_jobs, Args_NearestNeighbors, random_state=random_state)

    def prepare_export(self, n_components=None, random_state=None):
        """Select anchors and fit a fixed parameter-only export proposal.

        ``n_components=None`` selects all anchors. A smaller positive count
        selects distinct anchors uniformly without replacement. Every selected
        neighbourhood is found in the full source data.
        """
        n = len(self.data)
        if n_components is None:
            n_components = n
        if (isinstance(n_components, (bool, np.bool_)) or
                not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= n):
            raise ValueError("n_components must be an integer between 1 and the number of rows.")
        columns = list(self.data.columns)
        if any(not isinstance(name, str) for name in columns):
            raise ValueError("Model export requires string column names.")

        data = self.data.copy()
        if data.isna().any().any():
            data = fill_na_with_median(data, show_message=False)
        scale = data.std().fillna(0.0) if self.normalize else None
        if scale is not None:
            scale = scale.mask(scale == 0, 1)
        transformed = data / scale if scale is not None else data
        anchors = (np.arange(n) if n_components == n else
                   np.random.default_rng(random_state).choice(n, size=n_components, replace=False))
        k = min(self.K, n)
        if k == 1:
            neighbours = anchors[:, None]
        else:
            args = dict(self.Args_NearestNeighbors)
            args.pop("n_neighbors", None)
            search = NearestNeighbors(n_neighbors=k, **args).fit(transformed)
            neighbours = search.kneighbors(transformed.iloc[anchors], return_distance=False)

        means, covariances, rows = [], [], []
        for indices in neighbours:
            mean, covariance, row = _component_parameters(data.iloc[indices])
            means.append(mean)
            covariances.append(covariance)
            rows.append(row)
        model = ClusterCovSampler(means, covariances, np.full(n_components, 1.0 / n_components), columns)
        return _LocalCovExport(model, _diagnostics_frame(rows))


class clusterCov(clusterResampler):
    """`clusterCov` is a method for `clusterResampler` class to create 
    synthetic values from the multivariate normal distribution 
    with the covariance matrix estimated from the clusters.

    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param n_clusters: The number of clusters, defaults to 8
    :type n_clusters: int, optional
    :param size_min: Required minimum cluster size, defaults to None
    :type size_min: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    """    
    def __init__(self, data:DataFrame, n_clusters=8, size_min:int = None, normalize:bool = True, clipping:bool = True) -> None:
        super().__init__(data, method=self.method, n_clusters=n_clusters, size_min=size_min, normalize=normalize, clipping=clipping)
        self._model = None
        self._diagnostics = None

    def fit_model(self):
        """Fit raw-unit mixture parameters without generating synthetic records.

        Normalization affects clustering only; means and covariances are in
        original units. The legacy :meth:`fit` workflow remains unchanged.
        """
        self._model = None
        self._diagnostics = None
        current_data, _, _, labels, sizes = self._cluster_data()
        columns = list(current_data.columns)
        if any(not isinstance(name, str) for name in columns):
            raise ValueError("Model export requires string column names.")
        means, covariances, rows = [], [], []
        for label in sizes.index:
            mean, covariance, row = _component_parameters(current_data[labels == label])
            means.append(mean)
            covariances.append(covariance)
            rows.append(row)
        weights = sizes.to_numpy(dtype=float) / len(current_data)
        self._model = ClusterCovSampler(means, covariances, weights, columns)
        self._diagnostics = _diagnostics_frame(rows)
        return self

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("Call fit_model() before sampling, diagnostics or exporting.")
        return self._model

    @property
    def means_(self):
        return self._require_model().means_

    @property
    def covariances_(self):
        return self._require_model().covariances_

    @property
    def weights_(self):
        return self._require_model().weights_

    @property
    def columns(self):
        return self._require_model().columns

    def sample(self, n, random_state=None):
        """Sample fitted parameters only, without legacy source-bound clipping."""
        return self._require_model().sample(n, random_state=random_state)

    def disclosure_diagnostics(self):
        """Return per-component diagnostics computed from the source rows.

        One row per component, in the order of ``means_``: ``count`` (source
        rows), ``rank`` (rank of the centred rows), ``constant_columns``, and
        ``top1_share_<column>``/``top2_share_<column>`` (largest one and two
        values as shares of the component total; ``NaN`` for negative values or
        a zero total). The diagnostics are confidential, are not exported, apply
        no thresholds and do not establish privacy.
        """
        self._require_model()
        return self._diagnostics.copy()

    def export_model(self, path):
        """Write the fitted parameters to a versioned NPZ file."""
        self._require_model()._export(path)
        
    def method(self, cluster:DataFrame, size:int):
        """Creating synthetic values from the estimated multivariate normal distribution. 

        :param cluster: Cluster data
        :type cluster: pandas.DataFrame
        :param size: Required number of synthetic observations. Size is equal to the number of observations in the cluster if not specified.
        :type size: int
        :return: Synthetic values
        :rtype: pandas.DataFrame
        """        
        syn_values_np = _draw_local_normal(cluster, size=size)
        syn_values = DataFrame(syn_values_np, columns=cluster.columns)
        return syn_values
