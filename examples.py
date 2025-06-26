from synloc import kNNResampler
from synloc.tools import stochastic_rounder 
from mixedvines.mixedvine import MixedVine
import pandas as pd


class LocalMixedVine(kNNResampler):
    def __init__(self, data: pd.DataFrame, cont_cols: list, K: int = 30, normalize: bool = True, clipping: bool = True, Args_NearestNeighbors: dict = {}, random_state=None) -> None:
        self.cont_cols = cont_cols
        super().__init__(data, method=self.method, K=K, normalize=normalize, clipping=clipping, Args_NearestNeighbors=Args_NearestNeighbors, random_state=random_state)

    def method(self, data):
        # Defensive: check if data has enough rows/columns
        if data.shape[0] < 2:
            raise ValueError("Not enough data to fit MixedVine.")
        generator = MixedVine.fit(data.values, self.cont_cols)
        return generator.rvs(1)[0]