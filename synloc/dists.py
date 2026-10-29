from .kNNResampler import kNNResampler
from .clusterResampler import clusterResampler
from .clusterCovSampler import ClusterCovSampler
from copy import deepcopy
from scipy.stats import multivariate_normal
from pandas import DataFrame
import numpy as np

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

def _draw_local_normal(data, size=None):
    """Draw from local covariance while preserving locally constant columns."""
    means = data.mean(axis=0).to_numpy(dtype=float)
    constant_mask = _local_constant_mask(data)
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

    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, n_jobs:int=-1, Args_NearestNeighbors:dict={}):   
        super().__init__(data, self.method, K, normalize, clipping, n_jobs, Args_NearestNeighbors)


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
        super().__init__(data, method=self.method,n_clusters = n_clusters, size_min = size_min, normalize = normalize, clipping = clipping)
        self._model = None
        self._clearance = None

    def fit_model(self):
        """Fit raw-unit mixture parameters without generating synthetic records.

        Scaling affects clustering only. Contributor diagnostics assume one
        person per row; they do not establish privacy or regulatory clearance.
        The legacy :meth:`fit` workflow remains separate and unchanged.
        """
        self._model = None
        self._clearance = None
        current_data, _, _, labels, sizes = self._cluster_data()
        columns = list(current_data.columns)
        if any(not isinstance(name, str) for name in columns):
            raise ValueError("Model export requires string column names.")
        means, covariances, components = [], [], []
        for component, label in enumerate(sizes.index):
            cluster = current_data[labels == label]
            constant = _local_constant_mask(cluster)
            varying = np.flatnonzero(~constant)
            mean = cluster.mean().to_numpy(dtype=float, copy=True)
            mean[constant] = cluster.iloc[0].to_numpy(dtype=float)[constant]
            covariance = np.zeros((len(columns), len(columns)))
            if varying.size:
                covariance[np.ix_(varying, varying)] = _regularized_covariance(cluster.iloc[:, varying])
            means.append(mean)
            covariances.append(covariance)
            dominance = {}
            for name in columns:
                values = cluster[name].to_numpy(dtype=float)
                if np.any(values < 0):
                    dominance[name] = {"status": "not_applicable", "reason": "negative_values"}
                    continue
                maximum = float(values.max())
                if maximum == 0:
                    dominance[name] = {"status": "unassessed", "reason": "zero_total"}
                    continue
                # Scaling before summing avoids overflow without changing shares.
                shares = np.sort(values / maximum)
                total = float(shares.sum())
                top_one = float(shares[-1] / total)
                top_two = float(shares[-2:].sum() / total)
                dominance[name] = {
                    "status": "passed" if top_one <= 0.50 and top_two <= 0.67 else "failed",
                    "top_one_share": top_one,
                    "top_two_share": top_two,
                    "one_50_passed": bool(top_one <= 0.50),
                    "two_67_passed": bool(top_two <= 0.67),
                }
            components.append({
                "component": component,
                "count": int(len(cluster)),
                "empirical_rank": int(np.linalg.matrix_rank(cluster.to_numpy(dtype=float) - mean)),
                "constant_columns": [name for name, flag in zip(columns, constant) if flag],
                "dominance": dominance,
            })
        weights = sizes.to_numpy(dtype=float) / len(current_data)
        model = ClusterCovSampler(means, covariances, weights, columns)
        failed = [
            item["component"] for item in components
            if self.size_min is not None and item["count"] < self.size_min
        ]
        self._clearance = {
            "mechanical_validity": {"status": "passed", "schema_version": model.VERSION},
            "privacy_approved": False,
            "required_contributor_check": {
                "status": "failed" if failed else ("unassessed" if self.size_min is None else "passed"),
                "minimum_rows": self.size_min,
                "unit": "row; assumes one person per row",
                "failed_components": failed,
            },
            "research_export_eligible": not failed,
            "review_required": True,
            "components": components,
            "review_limitations": [
                "This is an internal diagnostic report, not regulatory clearance or a privacy guarantee.",
                "Contributor counts are rows, not verified unique persons; repeated persons require separate review.",
                "Household dependencies and linked records are not assessed by row counts.",
                "Covariance entries, cross-products and joint parameter disclosure require separate review.",
                "Mixture weights can leak cluster counts, especially when the source sample size is known.",
                "Nonnegative-variable 1/50 and 2/67 dominance checks are diagnostics only; negative values are not applicable and zero totals are unassessed.",
                "Constant columns, small or singular components, and repeated sampling can disclose fitted parameters.",
                "Parameter-only export and a reviewed acknowledgement do not authorize confidential-data release.",
                "This report contains source-derived counts and diagnostics and must not accompany an export without separate review.",
            ],
        }
        self._model = model
        self.means_ = model.means_
        self.covariances_ = model.covariances_
        self.weights_ = model.weights_
        self.columns = model.columns
        return self

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("Call fit_model() before sampling, reporting or exporting.")
        return self._model

    def sample(self, n, random_state=None):
        """Sample fitted parameters only, without legacy source-bound clipping."""
        return self._require_model().sample(n, random_state=random_state)

    def clearance_report(self):
        """Return internal JSON-serializable diagnostics, never release approval."""
        self._require_model()
        return deepcopy(self._clearance)

    def export_model(self, path, *, reviewed=False):
        """Write versioned parameter-only NPZ for an acknowledged research export.

        ``reviewed=True`` acknowledges unresolved privacy review, not approval.
        Failed required contributor checks always block export. Dominance and
        other unresolved diagnostics remain explicit in ``clearance_report``.
        """
        model = self._require_model()
        if self._clearance["required_contributor_check"]["status"] == "failed":
            raise ValueError("Export blocked by failed required contributor checks.")
        if reviewed is not True:
            raise ValueError("Export requires reviewed=True to acknowledge unresolved privacy review.")
        model._export(path)
        
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
