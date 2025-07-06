from .tools import fill_na_with_median, compareplots, stochastic_rounder, compute_k_distances
from sklearn.neighbors import NearestNeighbors
from pandas import DataFrame
import pandas as pd # Import pandas explicitly for DataFrame creation
from numpy import diag, sqrt, cov, random, concatenate, arange, array, vstack
from tqdm import tqdm
# Import joblib for parallel processing
from joblib import Parallel, delayed
import multiprocessing # To get CPU count

# Helper function for parallel processing (needs to be defined at the top level or static)
# It takes the necessary data explicitly to avoid issues with pickling 'self'
def _generate_one_synthetic_point(index, original_data, neighbors_indices, method_func):
    """
    Generates a single synthetic data point based on neighbors.
    Helper function for parallel execution.
    """
    matchindex = neighbors_indices[index, :]
    # Use iloc for potentially faster integer-based indexing
    subsample = original_data.iloc[matchindex]
    # The method should return a 1D array-like object (e.g., pd.Series, np.array)
    return method_func(subsample)


class kNNResampler(object):
    """Finds the nearest neighbor of each observation and creates synthetic values by the given method.

    :param data: Original data set to be synthesized
    :type data: pandas.DataFrame
    :param method: Function to be used to create synthetic values from each cluster.
                 Must accept a pandas.DataFrame (the neighbors) and return a
                 1D array-like object (e.g., pandas.Series, numpy.array) representing
                 the synthetic point.
    :type method: function
    :param K: The number of the nearest neighbors used to create synthetic samples, defaults to 30
    :type K: int, optional
    :param normalize: Normalize sample before defining clusters, defaults to True
    :type normalize: bool, optional
    :param clipping: trim values greater (smaller) than the maximum (minimum) for each variable, defaults to True
    :type clipping: bool, optional
    :param n_jobs: Number of CPU cores to use for parallel processing. -1 means using all processors. Defaults to -1.
    :type n_jobs: int, optional
    :param Args_NearestNeighbors: `NearestNeighbors` function arguments can be specified if needed. See scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html , defaults to {}
    :type Args_NearestNeighbors: dict, optional
    """
    def __init__ (self, data:DataFrame, method, K:int = 30, normalize:bool = True, clipping:bool = True, n_jobs:int = -1, Args_NearestNeighbors:dict = {}, random_state=None) -> None:
        """
        Initialize kNNResampler.
        :param random_state: Optional random seed for reproducibility.
        """
        self.data = data.copy().reset_index(drop = True)
        self.method = method
        self.K = K
        # assert that K must be greater tha n1
        if self.K < 1:
            raise ValueError("K must be greater than or equal to 1")
        self.normalize = normalize
        self.Args_NearestNeighbors = Args_NearestNeighbors
        self.clipping = clipping
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.fitted = False
        self.synthetic = None # Initialize synthetic attribute
        self._data_min = None # Store min/max for clipping
        self._data_max = None
        self.random_state = random_state

    def fit(self, sample_size = None) -> DataFrame:
        """Creating synthetic sample using parallel processing.

        :param sample_size: Required minimum size. The synthetic sample size will be the sample size if not specified., defaults to None
        :type sample_size: int, optional
        :return: Returns the synthetic sample
        :rtype: pandas.DataFrame
        """
        # Store parameters that will be needed in parallel processing
        if not hasattr(self, '_processing_params'):
            self._processing_params = {
                'method': self.method,
                'n_jobs': self.n_jobs if self.n_jobs != -1 else max(1, multiprocessing.cpu_count() // 2)  # Use half of available cores
            }

        ### Assertations
        if sample_size is not None:
            if not isinstance(sample_size, int) or sample_size <= 0:
                raise ValueError("sample_size must be a positive integer")

        ### Store original min/max for potential clipping later
        self._data_min = self.data.min()
        self._data_max = self.data.max()

        ### Checking/Imputing missing values
        # Operate on a temporary variable to avoid modifying self.data repeatedly if fit is called multiple times
        current_data = self.data.copy()
        if current_data.isna().any().any():
            print('The original sample has missing values. Missing values are replaced with variable medians.')
            # Ensure fill_na_with_median returns a DataFrame
            current_data = fill_na_with_median(current_data)
            # Update self.data only if imputation happened and it's the first fit?
            # Or maybe always work with the imputed version within fit? Let's use the imputed version.

        ### Normalizing data set
        if self.normalize:
            variances = current_data.var()
            if (variances == 0).any():
                print("Warning: Some columns are constant and will not be scaled.")
            variances[variances == 0] = 1
            scale_factors = sqrt(variances)
            dataN = current_data / scale_factors
        else:
            dataN = current_data.copy()
        # dataN is the normalized sample (or a copy) to calculate distances.

        ### Selecting index:
        n_original = current_data.shape[0]
        rng = random if self.random_state is None else __import__('numpy').random.RandomState(self.random_state)
        if sample_size is not None:
            actual_sample_size = sample_size
            selected_indices = rng.choice(n_original, actual_sample_size, replace=True)
        else:
            actual_sample_size = n_original
            selected_indices = arange(n_original)

        ### Nearest Neighbor Model
        if self.K > n_original:
            print(f"Warning: K={self.K} is greater than the number of data points ({n_original}). Setting K to {n_original}.")
            k_neighbors_count = n_original
        else:
            k_neighbors_count = self.K

        if k_neighbors_count == 1:
            neighbors_indices = arange(n_original).reshape(-1, 1)
        else:
            nn_to_find = k_neighbors_count - 1
            NNfit = NearestNeighbors(n_neighbors=nn_to_find, **self.Args_NearestNeighbors).fit(dataN)
            neighbor_indices_only = NNfit.kneighbors(dataN, return_distance=False)
            neighbors_indices = concatenate(
                (arange(n_original).reshape(-1, 1), neighbor_indices_only), axis=1
            )
            if neighbors_indices.shape[1] < k_neighbors_count:
                diff = k_neighbors_count - neighbors_indices.shape[1]
                padding = neighbors_indices[:, -1].reshape(-1,1)
                for _ in range(diff):
                    neighbors_indices = concatenate((neighbors_indices, padding), axis=1)
            neighbors_indices = neighbors_indices[:, :k_neighbors_count]

        ### Synthesizing using parallel processing
        print(f"Generating {actual_sample_size} synthetic samples using {self._processing_params['n_jobs']} cores...")
        
        # Process in batches to reduce DLL loading overhead
        batch_size = max(1, actual_sample_size // self._processing_params['n_jobs'])
        batches = [selected_indices[i:i + batch_size] for i in range(0, len(selected_indices), batch_size)]

        results_list = []
        try:
            with tqdm(total=actual_sample_size, desc="Generating synthetic samples") as pbar:
                with Parallel(n_jobs=self._processing_params['n_jobs'], prefer="processes") as parallel:
                    for batch in batches:
                        batch_results = parallel(
                            delayed(_generate_one_synthetic_point)(
                                idx,
                                current_data,
                                neighbors_indices,
                                self._processing_params['method']
                            ) for idx in batch
                        )
                        results_list.extend(batch_results)
                        pbar.update(len(batch_results))
        except Exception as e:
            raise RuntimeError(f"Parallel synthetic sample generation failed: {e}")

        # Check if results are valid before creating DataFrame
        if not results_list or results_list[0] is None:
            raise ValueError("Synthetic sample generation failed. The method function might not be returning expected values.")

        # Convert the list of results (Series/arrays) into a DataFrame
        # Using vstack might be slightly faster if results are numpy arrays
        try:
            synthetic_np = vstack(results_list)
            self.synthetic = DataFrame(synthetic_np, columns=current_data.columns)
        except Exception as e:
            print(f"Warning: Results from method function might not be uniform NumPy arrays. Creating DataFrame row by row. Error: {e}")
            self.synthetic = pd.concat([pd.Series(res) for res in results_list], axis=1).T
            self.synthetic.columns = current_data.columns
            self.synthetic.index = range(actual_sample_size)

        ### Clipping
        if self.clipping:
            # Use stored original min/max values for clipping
            # Ensure columns align
            for col in self.synthetic.columns:
                if col in self._data_min and col in self._data_max:
                    self.synthetic[col] = self.synthetic[col].clip(lower=self._data_min[col], upper=self._data_max[col])

        # Use the same K as for resampling
        self.data_distances = compute_k_distances(dataN, K=self.K)
        # For synthetic, normalize using the same scale_factors if normalization was applied
        if self.normalize:
            syntheticN = self.synthetic / scale_factors
        else:
            syntheticN = self.synthetic.copy()
        self.synthetic_distances = compute_k_distances(syntheticN, K=self.K)

        self.fitted = True
        print("Synthetic sample generation complete.")
        return self.synthetic

    def comparePlots(self, variable_list, fig_size = None):
        """Creating plots to compare the original sample and the synthetic sample.

        :param variable_list: A list of variables in the data set. The maximum list size must be 3. The type of the plot depends o the list size: 1->histogram, 2->scatter plot, 3->3D scatter plot.
        :type variable_list: list
        :param fig_size: The figure size can be adjusted, defaults to None
        :type fig_size: tuple, optional
        """
        if not self.fitted or self.synthetic is None:
            print("Model not fitted yet or synthetic data not generated. Call fit() first.")
            return
        # Compare using the original data passed to __init__
        compareplots(self.data, self.synthetic, variable = variable_list, fig_size = fig_size)

    def round_integers(self, integer_columns:list, stochastic:bool = True) -> None:
        """Rounds variables to integers. 

        :param integer_columns: The list of variables to be rounded.
        :type integer_columns: list
        :param stochastic: Variables are rounded by a stochastic process, defaults to True
        :type stochastic: bool, optional
        """        
        if self.fitted:
            if stochastic:
                self.synthetic.loc[:, integer_columns] = stochastic_rounder(self.synthetic.loc[:, integer_columns])
            else:
                self.synthetic.loc[:, integer_columns] = self.synthetic.loc[:, integer_columns].round()
        else: 
            print('The synthetic sample is not created yet. Use `.fit()` to create synthetic sample.')