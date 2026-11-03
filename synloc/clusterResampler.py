from .tools import (
    compareStats,
    compareplots,
    fill_na_with_median,
    quality_report,
    new_cluster_sizes,
    compute_k_distances,
    validate_numeric_dataframe,
)
from pandas import DataFrame, Series, concat
import numpy as np
from sklearn.cluster import KMeans

class clusterResampler(object):
    """Creating synthetic sample by clustering.

    This class creates subsamples from a given sample. 
    The subsamples are created by clustering the original sample and then 
    sampling from each cluster. The clustering is done by standard KMeans with a heuristic for size_min.

    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param method: Function to be used to create synthetic values from each cluster.
    :type method: function
    :param n_clusters: The number of clusters, defaults to 8
    :type n_clusters: int, optional
    :param size_min: Required minimum cluster size, defaults to None
    :type size_min: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    """
    def __init__ (self, data:DataFrame, method, n_clusters=8, size_min = None, normalize:bool = True, clipping:bool = True) -> None:

        self.data = validate_numeric_dataframe(data)
        self.method = method
        if size_min is not None and (
                isinstance(size_min, (bool, np.bool_)) or
                not isinstance(size_min, (int, np.integer)) or size_min <= 0):
            raise ValueError("size_min must be a positive integer or None.")
        if (isinstance(n_clusters, (bool, np.bool_)) or
                not isinstance(n_clusters, (int, np.integer)) or n_clusters <= 0):
            raise ValueError("n_clusters must be a positive integer.")
        self.size_min = int(size_min) if size_min is not None else None
        self.n_clusters = int(n_clusters)
        self.normalize = normalize
        self.clipping = clipping
        self.fitted = False
        self.synthetic = None
        self.metrics = None

    def _cluster_data(self):
        """Prepare data and final memberships shared by both fitting workflows."""

        current_data = self.data.copy()
        if current_data.isna().any().any():
            print('The original sample has missing values. Missing values are replaced with variable medians.')
            current_data = fill_na_with_median(current_data, show_message=False)
            # Public diagnostics compare against this same imputed reference.
            self.data = current_data.copy()

        scale = current_data.std().fillna(0.0) if self.normalize else None
        if scale is not None:
            scale = scale.mask(scale == 0, 1)
        dataN = current_data / scale if scale is not None else current_data

        
        ### Find clusters (Heuristic for size_min)
        n_original = current_data.shape[0]
        if self.size_min is not None and self.size_min > n_original:
            raise ValueError("size_min cannot be greater than the number of rows.")

        effective_n_clusters = min(self.n_clusters, n_original)
        if effective_n_clusters < self.n_clusters:
            print(
                f"Warning: n_clusters={self.n_clusters} is greater than the number "
                f"of data points ({n_original}). Setting n_clusters to {effective_n_clusters}."
            )

        kmeans = KMeans(n_clusters=effective_n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(dataN)

        # Heuristic: enforce size_min by reassigning points from small clusters
        if self.size_min is not None:
            labels = labels.copy()
            cluster_sizes = Series(labels).value_counts()
            small_clusters = cluster_sizes[cluster_sizes < self.size_min].index.tolist()
            large_clusters = cluster_sizes[cluster_sizes >= self.size_min].index.tolist()
            if not large_clusters and not cluster_sizes.empty:
                large_clusters = [int(cluster_sizes.idxmax())]
            if small_clusters:
                # Precompute cluster centers for large clusters
                centers = kmeans.cluster_centers_
                for sc in small_clusters:
                    if sc in large_clusters:
                        continue
                    idxs = (labels == sc).nonzero()[0]
                    for idx in idxs:
                        # Find nearest large cluster center
                        point = dataN.iloc[idx].values
                        dists = [((point - centers[lc])**2).sum() for lc in large_clusters]
                        nearest = large_clusters[int(np.argmin(dists))]
                        labels[idx] = nearest
                # Recompute cluster sizes after reassignment
                cluster_sizes = Series(labels).value_counts()
        else:
            cluster_sizes = Series(labels).value_counts()

        return current_data, dataN, scale, labels, cluster_sizes

    def fit(self, sample_size=None) -> DataFrame:
        """Create synthetic data, retaining legacy clipping and diagnostics.

        :param sample_size: Requested sample size, or the original row count.
        :return: The synthetic sample.
        :rtype: pandas.DataFrame
        """
        if sample_size is not None:
            if (isinstance(sample_size, (bool, np.bool_)) or
                    not isinstance(sample_size, (int, np.integer)) or sample_size <= 0):
                raise ValueError("sample_size must be a positive integer")
            sample_size = int(sample_size)

        self.fitted = False
        self.synthetic = None
        self.metrics = None
        self.data_distances = None
        self.synthetic_distances = None

        current_data, dataN, scale, labels, cluster_sizes = self._cluster_data()

        if sample_size is not None:
            cluster_sizes = new_cluster_sizes(cluster_sizes, sample_size)

        syn_samples = []
        for i in cluster_sizes.index:
            target_size = int(cluster_sizes.loc[i])
            if target_size <= 0:
                continue
            cluster_data = current_data[labels == i]
            if cluster_data.shape[0] == 0:
                continue
            syn_samples.append(self.method(cluster_data, target_size))

        if not syn_samples:
            raise ValueError("Synthetic sample generation failed because no non-empty clusters were available.")

        synthetic = concat(syn_samples, axis=0, ignore_index=True)
        ### Clipping
        if self.clipping:
            synthetic = synthetic.clip(lower=current_data.min(), upper=current_data.max(), axis=1)

        data_distances = compute_k_distances(dataN, K=self.size_min)
        synthetic_distances = compute_k_distances(synthetic / scale if scale is not None else synthetic, K=self.size_min)
        metrics = quality_report(current_data, synthetic)

        self.synthetic = synthetic
        self.data_distances = data_distances
        self.synthetic_distances = synthetic_distances
        self.metrics = metrics

        self.fitted = True
        return self.synthetic

    def comparePlots(self, variable_list, fig_size = None):
        """Creating plots to compare the original sample and the synthetic sample.

        :param variable_list: A list of variables in the data set. The maximum list size must be 3. The type of the plot depends o the list size: 1->histogram, 2->scatter plot, 3->3D scatter plot. 
        :type variable_list: list
        :param fig_size: The figure size can be adjusted, defaults to None
        :type fig_size: tuple, optional
        """        
        if not self.fitted or self.synthetic is None:
            print("Model not fitted yet or synthetic data not generated. Call fit() first.")
            return
        compareplots(self.data, self.synthetic, variable = variable_list, fig_size = fig_size)

    def compareStats(self):
        """Return variable-level quality metrics for the synthetic sample."""
        if not self.fitted or self.synthetic is None:
            print("Model not fitted yet or synthetic data not generated. Call fit() first.")
            return None
        return compareStats(self.data, self.synthetic)

    def qualityReport(self):
        """Return per-variable and overall quality metrics for the synthetic sample."""
        if not self.fitted or self.synthetic is None:
            print("Model not fitted yet or synthetic data not generated. Call fit() first.")
            return None
        return quality_report(self.data, self.synthetic)
