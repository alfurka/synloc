from .tools import fill_na_with_median, compareplots, new_cluster_sizes, compute_k_distances
from pandas import DataFrame, Series, concat
from numpy import diag, sqrt, cov
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

        self.data = data.reset_index(drop = True)
        self.method = method
        self.size_min = size_min
        self.n_clusters = n_clusters
        self.normalize = normalize
        self.clipping = clipping
        self.fitted = False
    def fit(self, sample_size = None) -> DataFrame:
        """Creating synthetic sample.

        :param sample_size: Required minimum size. The synthetic sample size will be the cluster size if not specified., defaults to None
        :type sample_size: int, optional
        :return: Returns the synthetic sample
        :rtype: pandas.DataFrame
        """        
        ### Assertations
        if sample_size is not None:
            assert type(sample_size) is int
            assert sample_size > 0

        ### Checking/Imputing missing values 

        if self.data.isna().any().any():
            print('The original sample has missing values. Missing values are replaced with variable medians.')
            self.data = fill_na_with_median(self.data)

        ### Normalizing data set

        if self.normalize:
            varMatrix = diag(cov(self.data.T)).copy()
            varMatrix[varMatrix==0] = 1 # don't do normalization if the variance is zero.
            dataN = self.data / sqrt(varMatrix)
        else: 
            dataN = self.data 
        # dataN is the normalized sample to calculate distances - if normalize == True.

        
        ### Find clusters (Heuristic for size_min)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        labels = kmeans.fit_predict(dataN)

        # Heuristic: enforce size_min by reassigning points from small clusters
        if self.size_min is not None:
            labels = labels.copy()
            cluster_sizes = Series(labels).value_counts()
            small_clusters = cluster_sizes[cluster_sizes < self.size_min].index.tolist()
            large_clusters = cluster_sizes[cluster_sizes >= self.size_min].index.tolist()
            if small_clusters:
                # Precompute cluster centers for large clusters
                centers = kmeans.cluster_centers_
                for sc in small_clusters:
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
            syn_samples.append(self.method(self.data[labels == i], cluster_sizes[i]))

        self.synthetic = concat(syn_samples, axis=0)
        ### Clipping
        if self.clipping:
            self.synthetic = self.synthetic.clip(lower=self.data.min(), upper=self.data.max(), axis=1)

        # Use the same normalization as above
        
        self.data_distances = compute_k_distances(dataN, K=self.size_min)
        # For synthetic, normalize using the same varMatrix if normalization was applied
        if self.normalize:
            syntheticN = self.synthetic / sqrt(varMatrix)
        else:
            syntheticN = self.synthetic.copy()
        self.synthetic_distances = compute_k_distances(syntheticN, K=self.size_min)

        self.fitted = True
        return self.synthetic

    def comparePlots(self, variable_list, fig_size = None):
        """Creating plots to compare the original sample and the synthetic sample.

        :param variable_list: A list of variables in the data set. The maximum list size must be 3. The type of the plot depends o the list size: 1->histogram, 2->scatter plot, 3->3D scatter plot. 
        :type variable_list: list
        :param fig_size: The figure size can be adjusted, defaults to None
        :type fig_size: tuple, optional
        """        
        compareplots(self.data, self.synthetic, variable = variable_list, fig_size = fig_size)