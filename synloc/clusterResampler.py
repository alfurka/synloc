from .tools import fill_na_with_median, compareplots, new_cluster_sizes
from pandas import DataFrame, Series, concat
from numpy import diag, sqrt, cov
from k_means_constrained import KMeansConstrained

class clusterResampler(object):
    """Creating synthetic sample by clusterig.

    This class creates subsamples from a given sample. 
    The subsamples are created by clustering the original sample and then 
    sampling from each cluster. The clustering is done by the `KMeansConstrained` 
    function from `k-means-constrained` package.

    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param method: Function to be used to create synthetic values from each cluster.
    :type method: function
    :param n_clusters: The number of clusters, defaults to 8
    :type n_clusters: int, optional
    :param size_min: Required minimum cluster size, defaults to None
    :type size_min: int, optional
    :param size_max: Required maximum cluster size, defaults to None
    :type size_max: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    """
    def __init__ (self, data:DataFrame, method, n_clusters=8, size_min = None, size_max = None, normalize:bool = True, clipping:bool = True) -> None: 

        self.data = data.reset_index(drop = True)
        self.method = method
        self.size_min = size_min
        self.n_clusters = n_clusters
        self.size_max = size_max
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

        
        ### Find clusters

        clf = KMeansConstrained(n_clusters=self.n_clusters, size_min = self.size_min, size_max=self.size_max)
        clf.fit_predict(dataN)
        
        ### Selecting index:

        cluster_sizes = Series(clf.labels_).value_counts()

        if sample_size is not None:
            cluster_sizes = new_cluster_sizes(cluster_sizes, sample_size)


        syn_samples = []
        for i in range(cluster_sizes.shape[0]):
            syn_samples.append(self.method(self.data[clf.labels_ == i], cluster_sizes[i]))
        
        self.synthetic = concat(syn_samples, axis = 0)
        ### Clipping

        if self.clipping:
            self.synthetic = self.synthetic.clip(lower=self.data.min(), upper=self.data.max(), axis = 1)

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