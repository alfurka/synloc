from .kNNResampler import kNNResampler
from .clusterResampler import clusterResampler
from .tools import stochastic_rounder
from scipy.stats import multivariate_normal
from pandas import DataFrame
from synthia import FPCADataGenerator, CopulaDataGenerator, GaussianCopula 

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
    :param Args_NearestNeighbors: `NearestNeighbors` function arguments can be specified if needed. See scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html , defaults to {}
    :type Args_NearestNeighbors: dict, optional
    """    
    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}):   
        super().__init__(data, self.method, K, normalize, clipping, Args_NearestNeighbors)
        
    def round_integers(self, integer_columns:list, stochastic:bool = True) -> None:
        """Rounds variables to integers. 

        :param integer_columns: The list of variables to be rounded.
        :type integer_columns: list
        :param stochastic: Variables are rounded by a stochastic process, defaults to True
        :type stochastic: bool, optional
        """        
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')

    def method(self, subsample:DataFrame):
        """Estimates covariance matrix and draw samples from the estimated multivariate normal distribution.

        :param subsample: A subsample defined by the `kNNResampler` class.
        :type subsample: pandas.DataFrame
        :return: Synthetic values. 
        :rtype: numpy.darray
        """        
        covMatrix = subsample.cov()
        return subsample.mean(0) + multivariate_normal.rvs(cov = covMatrix)

class LocalFPCA(kNNResampler):
    """It is a method for `kNNResampler` class. The method is based on the 
    `FPCADataGenerator` class from the `synthia` package. 
    
    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param n_fpca_components: The number of dimensions after PCA, defaults to 2
    :type n_fpca_components: int, optional
    :param K: The number of the nearest neighbors used to create synthetic samples, defaults to 30
    :type K: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    :param Args_NearestNeighbors: `NearestNeighbors` function arguments can be specified if needed. See scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html , defaults to {}
    :type Args_NearestNeighbors: dict, optional
    """    
    def __init__(self, data:DataFrame, n_fpca_components:int=2, K: int = 30, normalize: bool = True, clipping: bool = True, Args_NearestNeighbors: dict = {}) -> None:        
        super().__init__(data, self.method, K, normalize, clipping, Args_NearestNeighbors)
        self.n_fpca_components = n_fpca_components
    def round_integers(self, integer_columns:list, stochastic:bool = True):
        """Rounds variables to integers. 

        :param integer_columns: The list of variables to be rounded.
        :type integer_columns: list
        :param stochastic: Variables are rounded by a stochastic process, defaults to True
        :type stochastic: bool, optional
        """  
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')
    def method(self, data):
        """Creates syntehtic values using `FPCADataGenerator` class from the `synthia` package.

        :param data: A subsample defined by the `kNNResampler` class.
        :type data: pandas.DataFrame
        :return: Synthetic values. 
        :rtype: numpy.darray
        """
        generator = FPCADataGenerator()
        generator.fit(data, n_fpca_components=self.n_fpca_components)
        return generator.generate(1)[0]

class LocalGaussianCopula(kNNResampler):
    """It is a method for `kNNResampler` class to create synthetic 
    values using gaussian copula.


    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param K: The number of the nearest neighbors used to create synthetic samples, defaults to 30
    :type K: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    :param Args_NearestNeighbors: `NearestNeighbors` function arguments can be specified if needed. See scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html , defaults to {}
    :type Args_NearestNeighbors: dict, optional
    """    
    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}):
        super().__init__(data, self.method, K, normalize, clipping, Args_NearestNeighbors)
    
    def round_integers(self, integer_columns:list, stochastic:bool = True):
        """Rounds variables to integers. 

        :param integer_columns: The list of variables to be rounded.
        :type integer_columns: list
        :param stochastic: Variables are rounded by a stochastic process, defaults to True
        :type stochastic: bool, optional
        """ 
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')
    
    def method(self, subsample:DataFrame):
        """Creating synthetic values using Gaussian copula.

        :param subsample: A subsample defined by the `kNNResampler` class.
        :type subsample: pandas.DataFrame
        :return: Synthetic values. 
        :rtype: numpy.darray
        """  
        generator = CopulaDataGenerator()
        generator.fit(subsample, copula=GaussianCopula() ,parameterize_by=None)
        return generator.generate(1)[0]


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
    :param size_max: Required maximum cluster size, defaults to None
    :type size_max: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    """    
    def __init__(self, data:DataFrame, n_clusters=8, size_min:int = None, size_max:int=None, normalize:bool = True, clipping:bool = True) -> None:        
        super().__init__(data, method=self.method,n_clusters = n_clusters, size_min = size_min, size_max = size_max, normalize = normalize, clipping = clipping)
        
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
        syn_values = DataFrame(multivariate_normal.rvs(cov = covMatrix, size = size))
        syn_values.columns = cluster.columns
        syn_values = syn_values.add(cluster.mean(0))
        return syn_values 

class clusterGaussCopula(clusterResampler):
    """`clusterGaussCopula` is a method for `clusterResampler` class to 
    create synthetic values from Gaussian copula. 
    
    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
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
    def __init__(self, data:DataFrame, n_clusters =8, size_min:int = None, size_max:int=None, normalize:bool = True, clipping:bool = True) -> None:
        """C
        """  
        super().__init__(data, method=self.method, n_clusters=n_clusters,size_min = size_min, size_max = size_max, normalize = normalize, clipping = clipping)
        
    def method(self, cluster:DataFrame, size:int):
        """Creating synthetic values from Gaussian copula. 

        :param cluster: Cluster data
        :type cluster: pandas.DataFrame
        :param size: Required number of synthetic observations. Size is equal to the number of observations in the cluster if not specified.
        :type size: int
        :return: Synthetic values
        :rtype: pandas.DataFrame
        """  
        generator = CopulaDataGenerator()
        generator.fit(cluster, copula=GaussianCopula() ,parameterize_by=None)
        syn_sample = DataFrame(generator.generate(size))
        syn_sample.columns = cluster.columns
        return syn_sample