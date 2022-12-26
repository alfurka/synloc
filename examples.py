from synloc import kNNResampler
from synloc.tools import stochastic_rounder 
from mixedvines.mixedvine import MixedVine
import pandas as pd


class LocalMixedVine(kNNResampler):
    def __init__(self, data: pd.DataFrame, cont_cols:list, K: int = 30, normalize: bool = True, clipping: bool = True, Args_NearestNeighbors: dict = {}) -> None:
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method = self.method)
        self.cont_cols = cont_cols
    
    def method(self, data):
        generator = MixedVine.fit(data.values, self.cont_cols)
        return generator.rvs(1)[0]