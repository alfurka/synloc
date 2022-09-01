from .LocalResampler import LocalResampler
from .tools import *
from scipy.stats import multivariate_normal

class LocalCov(LocalResampler):
    def __init__(self, data:pd.DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}) -> None:
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method=self.method)
        
    def round_integers(self, integer_columns:list, stochastic:bool = True) -> None:
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')

    def method(self, subsample:pd.DataFrame):
        covMatrix = subsample.cov()
        return subsample.mean(0) + multivariate_normal.rvs(cov = covMatrix)    