from .tools import fill_na_with_median, compareplots, new_cluster_sizes
from .dists import *
from pandas import DataFrame, Series, concat
from numpy import diag, sqrt, cov
from k_means_constrained import KMeansConstrained

class clusterResampler(object):
    def __init__ (self, data:DataFrame, method, n_clusters=8, size_min = None, size_max = None, normalize:bool = True, clipping:bool = True) -> None: 
        ### Initializing 

        self.data = data.reset_index(drop = True)
        self.method = method
        self.size_min = size_min
        self.n_clusters = n_clusters
        self.size_max = size_max
        self.normalize = normalize
        self.clipping = clipping
        self.fitted = False
    def fit(self, sample_size = None) -> DataFrame:
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
        compareplots(self.data, self.synthetic, variable = variable_list, fig_size = fig_size)