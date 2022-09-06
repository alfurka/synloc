from .tools import *
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from tqdm import tqdm

class kNNResampler(object):
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

