from .kNNResampler import kNNResampler
from .clusterResampler import clusterResampler
from .tools import stochastic_rounder
from scipy.stats import multivariate_normal
from pandas import DataFrame
from synthia import FPCADataGenerator, CopulaDataGenerator, GaussianCopula 

class LocalCov(kNNResampler):
    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}):
        super().__init__(data, self.method, K, normalize, clipping, Args_NearestNeighbors)
        
    def round_integers(self, integer_columns:list, stochastic:bool = True) -> None:
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')

    def method(self, subsample:DataFrame):
        covMatrix = subsample.cov()
        return subsample.mean(0) + multivariate_normal.rvs(cov = covMatrix)

class LocalFPCA(kNNResampler):
    def __init__(self, data:DataFrame,n_fpca_components:int=2, K: int = 30, normalize: bool = True, clipping: bool = True, Args_NearestNeighbors: dict = {}) -> None:
        super().__init__(data, self.method, K, normalize, clipping, Args_NearestNeighbors)
        self.n_fpca_components = n_fpca_components
    def round_integers(self, integer_columns:list, stochastic:bool = True):    
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')
    def method(self, data):
        generator = FPCADataGenerator()
        generator.fit(data, n_fpca_components=self.n_fpca_components)
        return generator.generate(1)[0]

class LocalGaussianCopula(kNNResampler):
    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}):
        super().__init__(data, self.method, K, normalize, clipping, Args_NearestNeighbors)
    
    def round_integers(self, integer_columns:list, stochastic:bool = True):    
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')
    
    def method(self, subsample:DataFrame):
        generator = CopulaDataGenerator()
        generator.fit(subsample, copula=GaussianCopula() ,parameterize_by=None)
        return generator.generate(1)[0]


class clusterCov(clusterResampler):
    def __init__(self, data:DataFrame, n_clusters=8, size_min:int = None, size_max:int=None, normalize:bool = True, clipping:bool = True) -> None:
        super().__init__(data, method=self.method,n_clusters = n_clusters, size_min = size_min, size_max = size_max, normalize = normalize, clipping = clipping)
        
    def method(self, cluster:DataFrame, size:int):
        covMatrix = cluster.cov()
        syn_values = DataFrame(multivariate_normal.rvs(cov = covMatrix, size = size))
        syn_values.columns = cluster.columns
        syn_values = syn_values.add(cluster.mean(0))
        return syn_values 

class clusterGaussCopula(clusterResampler):
    def __init__(self, data:DataFrame, n_clusters =8, size_min:int = None, size_max:int=None, normalize:bool = True, clipping:bool = True) -> None:
        super().__init__(data, method=self.method, n_clusters=n_clusters,size_min = size_min, size_max = size_max, normalize = normalize, clipping = clipping)
        
    def method(self, cluster:DataFrame, size:int):
        generator = CopulaDataGenerator()
        generator.fit(cluster, copula=GaussianCopula() ,parameterize_by=None)
        syn_sample = DataFrame(generator.generate(size))
        syn_sample.columns = cluster.columns
        return syn_sample