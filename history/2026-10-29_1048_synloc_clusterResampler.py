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
from numpy import sqrt
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
        self.size_min = size_min
        if size_min is not None and (not isinstance(size_min, int) or size_min <= 0):
            raise ValueError("size_min must be a positive integer or None.")
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        self.n_clusters = n_clusters
        self.normalize = normalize
        self.clipping = clipping
        self.fitted = False
        self.synthetic = None
        self.metrics = None

    def fit(self, sample_size = None) -> DataFrame:
        """Creating synthetic sample.

        :param sample_size: Required minimum size. The synthetic sample size will be the cluster size if not specified., defaults to None
        :type sample_size: int, optional
        :return: Returns the synthetic sample
        :rtype: pandas.DataFrame
        """        
        ### Assertations
        if sample_size is not None:
            if not isinstance(sample_size, int) or sample_size <= 0:
                raise ValueError("sample_size must be a positive integer")

        ### Checking/Imputing missing values 

        current_data = self.data.copy()
        if current_data.isna().any().any():
            print('The original sample has missing values. Missing values are replaced with variable medians.')
            current_data = fill_na_with_median(current_data, show_message=False)
            self.data = current_data.copy()

        ### Normalizing data set

        if self.normalize:
            variances = current_data.var().fillna(0.0)
            variances[variances == 0] = 1 # don't do normalization if the variance is zero.
            dataN = current_data / sqrt(variances)
        else: 
            dataN = current_data.copy()
        # dataN is the normalized sample to calculate distances - if normalize == True.

        
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

        self.synthetic = concat(syn_samples, axis=0, ignore_index=True)
        ### Clipping
        if self.clipping:
            self.synthetic = self.synthetic.clip(lower=current_data.min(), upper=current_data.max(), axis=1)

        # Use the same normalization as above
        
        self.data_distances = compute_k_distances(dataN, K=self.size_min)
        # For synthetic, normalize using the same varMatrix if normalization was applied
        if self.normalize:
            syntheticN = self.synthetic / sqrt(variances)
        else:
            syntheticN = self.synthetic.copy()
        self.synthetic_distances = compute_k_distances(syntheticN, K=self.size_min)
        self.metrics = quality_report(current_data, self.synthetic)

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
