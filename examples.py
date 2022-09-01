from synloc import kNNResampler
from synloc.tools import stochastic_rounder 
from mixedvines.mixedvine import MixedVine
from synthia import FPCADataGenerator
import pandas as pd


class LocalMixedVine(kNNResampler):
    def __init__(self, data: pd.DataFrame, cont_cols:list, K: int = 30, normalize: bool = True, clipping: bool = True, Args_NearestNeighbors: dict = {}) -> None:
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method = self.method)
        self.cont_cols = cont_cols
    
    def method(self, data):
        generator = MixedVine.fit(data.values, self.cont_cols)
        return generator.rvs(1)[0]

class LocalFPCA(kNNResampler):
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