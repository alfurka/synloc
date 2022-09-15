from .kNNResampler import kNNResampler
from .tools import stochastic_rounder
from scipy.stats import multivariate_normal
from synthia import CopulaDataGenerator, GaussianCopula
from pandas import DataFrame

class LocalCov(kNNResampler):
    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}) -> None:
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method=self.method)
        
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

class LocalGaussianCopula(kNNResampler):
    def __init__(self, data:DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}):
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method=self.method)
    
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