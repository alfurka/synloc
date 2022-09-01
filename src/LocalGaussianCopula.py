from .LocalResampler import LocalResampler
from .tools import *
from synthia import GaussianCopula, CopulaDataGenerator


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