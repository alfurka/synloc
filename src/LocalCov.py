from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from synpytools import *
from synthia import GaussianCopula, CopulaDataGenerator, FPCADataGenerator
from tqdm import tqdm
from mixedvines.mixedvine import MixedVine


class LocalResampler(object):
    def __init__ (self, data:pd.DataFrame, K:int = 30, normalize:bool = True, clipping:bool = True, Args_NearestNeighbors:dict = {}, method = 'normal') -> None: 
        ### Initializing 

        self.data = data.reset_index(drop = True)
        self.method = method
        self.K = K
        self.normalize = normalize
        self.Args_NearestNeighbors = Args_NearestNeighbors
        self.clipping = clipping
        self.fitted = False
    def fit(self, sample_size = None) -> pd.DataFrame:
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
            varMatrix = np.diag(np.cov(self.data.T))
            dataN = self.data / np.sqrt(varMatrix)
        else: 
            dataN = self.data 
        # dataN is the normalized sample to calculate distances - if normalize == True.

        ### Selecting index:

        if sample_size is not None:
            selected_index = np.random.choice(dataN.index, sample_size)
        else:
            selected_index = dataN.index
            sample_size = self.data.shape[0]
        
        ### Nearest Neighbor Model

        NNfit = NearestNeighbors(n_neighbors = self.K - 1, **self.Args_NearestNeighbors).fit(dataN)
        neighbors = np.concatenate(
                (np.arange(self.data.shape[0]).reshape(-1,1), 
                    NNfit.kneighbors(return_distance=False)), axis = 1)
        
        ### Synthetizing...
        self.synthetic = self.data.loc[selected_index].copy()
        self.synthetic = self.synthetic.reset_index(drop = True)
        
        for i in tqdm(range(sample_size)):
            matchindex = neighbors[selected_index[i],:]
            self.synthetic.loc[i] = self.method(self.data.loc[matchindex])

        ### Clipping

        if self.clipping:
            self.synthetic = self.synthetic.clip(lower=self.data.min(), upper=self.data.max(), axis = 1)

        self.fitted = True
        return self.synthetic

    def comparePlots(self, variable_list, fig_size = None):
        compareplots(self.data, self.synthetic, variable = variable_list, fig_size = fig_size)


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

class LocalGaussianCopula(LocalResampler):
    def __init__(self, data:pd.DataFrame, K:int=30, normalize:bool=True, clipping:bool=True, Args_NearestNeighbors:dict={}):
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method=self.method)
    
    def round_integers(self, integer_columns:list, stochastic:bool = True):    
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')
    def method(self, subsample:pd.DataFrame):
        generator = CopulaDataGenerator()
        generator.fit(subsample, copula=GaussianCopula() ,parameterize_by=None)
        return generator.generate(1)[0]

class LocalMixedVine(LocalResampler):
    def __init__(self, data: pd.DataFrame, cont_cols:list, K: int = 30, normalize: bool = True, clipping: bool = True, Args_NearestNeighbors: dict = {}, method='normal') -> None:
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method = self.method)
        self.cont_cols = cont_cols
    
    def method(self, data):
        generator = MixedVine.fit(data.values, self.cont_cols)
        return generator.rvs(1)[0]

class LocalFPCA(LocalResampler):
    def __init__(self, data: pd.DataFrame, K: int = 30, normalize: bool = True, clipping: bool = True, Args_NearestNeighbors: dict = {}) -> None:
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method = self.method)
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
        generator.fit(data, n_fpca_components=2)
        return generator.generate(1)[0]




if __name__ == "__main__":
    from copulas.datasets import sample_trivariate_xyz
    k = sample_trivariate_xyz()
    K_set = 25

    ss_cop = LocalFPCA(K = K_set, data = k)
    ss_cov = LocalCov(K = K_set, data = k)
    
    dfSyn_cop = ss_cop.fit(sample_size=None)
    dfSyn_cov = ss_cov.fit(sample_size=None)

    ss_cop.comparePlots(['x','y','z'])
    ss_cov.comparePlots(['x','y','z'])