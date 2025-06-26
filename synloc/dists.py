from .kNNResampler import kNNResampler
from .clusterResampler import clusterResampler
from scipy.stats import multivariate_normal
from pandas import DataFrame

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
        covMatrix = subsample.cov()
        mean_vector = subsample.mean(0)
        return multivariate_normal.rvs(mean=mean_vector, cov=covMatrix)

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
        
    def method(self, cluster:DataFrame, size:int):
        """Creating synthetic values from the estimated multivariate normal distribution. 

        :param cluster: Cluster data
        :type cluster: pandas.DataFrame
        :param size: Required number of synthetic observations. Size is equal to the number of observations in the cluster if not specified.
        :type size: int
        :return: Synthetic values
        :rtype: pandas.DataFrame
        """        
        covMatrix = cluster.cov()
        mean_vector = cluster.mean(0)
        syn_values_np = multivariate_normal.rvs(mean=mean_vector, cov=covMatrix, size=size)
        syn_values = DataFrame(syn_values_np, columns=cluster.columns)
        return syn_values
