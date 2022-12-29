from .tools import fill_na_with_median, compareplots
from sklearn.neighbors import NearestNeighbors
from pandas import DataFrame
from numpy import diag, sqrt, cov, random, concatenate, arange
from tqdm import tqdm

class kNNResampler(object):
    """Finds the nearest neighbor of each observation and creates synthetic values by the given method. 
    
    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param method: Function to be used to create synthetic values from each cluster.
    :type method: function
    :param K: The number of the nearest neighbors used to create synthetic samples, defaults to 30
    :type K: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    :param Args_NearestNeighbors: `NearestNeighbors` function arguments can be specified if needed. See scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html , defaults to {}
    :type Args_NearestNeighbors: dict, optional
    """        
    def __init__ (self, data:DataFrame, method, K:int = 30, normalize:bool = True, clipping:bool = True, Args_NearestNeighbors:dict = {}) -> None: 
        self.data = data.reset_index(drop = True)
        self.method = method
        self.K = K
        self.normalize = normalize
        self.Args_NearestNeighbors = Args_NearestNeighbors
        self.clipping = clipping
        self.fitted = False
    def fit(self, sample_size = None) -> DataFrame:
        """Creating synthetic sample.

        :param sample_size: Required minimum size. The synthetic sample size will be the sample size if not specified., defaults to None
        :type sample_size: int, optional
        :return: Returns the synthetic sample
        :rtype: pandas.DataFrame
        """ 
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
            varMatrix = diag(cov(self.data.T)).copy()
            varMatrix[varMatrix==0] = 1 # don't do normalization if the variance is zero.
            dataN = self.data / sqrt(varMatrix)
        else: 
            dataN = self.data 
        # dataN is the normalized sample to calculate distances - if normalize == True.

        ### Selecting index:

        if sample_size is not None:
            selected_index = random.choice(dataN.index, sample_size)
        else:
            selected_index = dataN.index
            sample_size = self.data.shape[0]
        
        ### Nearest Neighbor Model

        NNfit = NearestNeighbors(n_neighbors = self.K - 1, **self.Args_NearestNeighbors).fit(dataN)
        neighbors = concatenate(
                (arange(self.data.shape[0]).reshape(-1,1), 
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
        """Creating plots to compare the original sample and the synthetic sample.

        :param variable_list: A list of variables in the data set. The maximum list size must be 3. The type of the plot depends o the list size: 1->histogram, 2->scatter plot, 3->3D scatter plot. 
        :type variable_list: list
        :param fig_size: The figure size can be adjusted, defaults to None
        :type fig_size: tuple, optional
        """ 
        compareplots(self.data, self.synthetic, variable = variable_list, fig_size = fig_size)