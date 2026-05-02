import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from numpy import random, floor, sin, cos, pi, square
import matplotlib.pyplot as plt
from pandas import DataFrame, Series

def validate_numeric_dataframe(dataframe, name="data"):
    """Validate and return a clean numeric DataFrame copy.

    ``synloc`` expects tabular numeric data. Categorical variables should be
    encoded before calling the resamplers. Boolean dummy columns are accepted
    and converted to 0/1 numeric values.
    """
    if not isinstance(dataframe, DataFrame):
        raise TypeError(f"{name} must be a pandas.DataFrame.")

    if dataframe.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if dataframe.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    if dataframe.columns.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate column names.")

    cleaned = dataframe.copy().reset_index(drop=True)
    non_numeric = []
    for column in cleaned.columns:
        dtype = cleaned[column].dtype
        if is_bool_dtype(dtype):
            cleaned[column] = cleaned[column].astype(int)
        elif not is_numeric_dtype(dtype):
            non_numeric.append(column)

    if non_numeric:
        cols = ", ".join(map(str, non_numeric))
        raise TypeError(
            f"{name} contains non-numeric columns: {cols}. "
            "Encode categorical variables as numeric dummy columns before using synloc."
        )

    cleaned = cleaned.apply(pd.to_numeric, errors="raise")
    values = cleaned.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain positive or negative infinity.")
    if cleaned.isna().all(axis=0).any():
        cols = ", ".join(map(str, cleaned.columns[cleaned.isna().all(axis=0)]))
        raise ValueError(
            f"{name} contains columns with only missing values: {cols}. "
            "At least one observed numeric value is required per column."
        )

    return cleaned

def fill_na_with_median(dataframe, show_message=True):
    dataframe = validate_numeric_dataframe(dataframe, name="dataframe")
    medians = dataframe.median(axis=0)
    if medians.isna().any():
        cols = ", ".join(map(str, medians.index[medians.isna()]))
        raise ValueError(f"Cannot impute columns with no observed values: {cols}.")
    dataframe = dataframe.fillna(medians)
    dataframe = dataframe.reset_index(drop=True)
    if show_message:
        print('Missing values are filled with variable medians.')
    return dataframe

def stochastic_rounder(x):
    """
    Rounds a float to an integer based on a stochastic process.
    A value of 5.3 has a 30% chance of being rounded to 6 and a 70% chance of being rounded to 5.
    """
    lower_int = floor(x)
    prob = x - lower_int
    return lower_int + random.binomial(1, prob, size=x.shape)

def stochastic_up_or_down(dataframe, p):
    up_or_down = 2 * (random.binomial(1, 0.5, dataframe.shape) - 0.5)
    outcomes = random.binomial(1, p, dataframe.shape)
    new_dataframe = dataframe + up_or_down * outcomes
    return new_dataframe.clip(lower=dataframe.min(axis=0), upper=dataframe.max(axis=0), axis = 1)

def compareplots(original_data, syn_data, variable, fig_size = (10,8)):
    original_data = validate_numeric_dataframe(original_data, name="original_data")
    syn_data = validate_numeric_dataframe(syn_data, name="syn_data")

    if isinstance(variable, str):
        variables = [variable]
    else:
        variables = list(variable)

    if len(variables) == 0:
        raise ValueError("variable must contain at least one column name.")

    missing = [column for column in variables if column not in original_data.columns]
    if missing:
        raise ValueError(f"Unknown column(s) in original_data: {missing}.")
    missing = [column for column in variables if column not in syn_data.columns]
    if missing:
        raise ValueError(f"Unknown column(s) in syn_data: {missing}.")

    if len(variables) == 1: # Histogram
        variable = variables[0]
        if fig_size is None:
            fig_size = (10, 8)
        plt.figure(figsize = fig_size)
        plt.title('Original and Synthetic values of variable `{}`'.format(variable))
        plt.hist(original_data[variable], alpha = 0.5, label = 'Original Sample')
        plt.hist(syn_data[variable], alpha = 0.5, label = 'Synthetic Sample')
        plt.legend(loc='upper right')
        plt.show()
    else:
        if len(variables) == 2: # Scatter plot
            if fig_size is None:
                fig_size = (15,7)          
            fig = plt.figure(figsize = fig_size)
            ax1 = fig.add_subplot(121)
            ax1.set_title('Original sample: `{}` and `{}`'.format(*variables))
            ax1.scatter(original_data[variables[0]] , original_data[variables[1]])
            ax2 = fig.add_subplot(122)
            ax2.set_title('Synthetic sample: `{}` and `{}`'.format(*variables))
            ax2.scatter(syn_data[variables[0]] , syn_data[variables[1]])
            plt.show()
        
        elif len(variables) == 3: # 3d scatter plot
            if fig_size is None:
                fig_size = (15,7) 
            fig = plt.figure(figsize = fig_size)
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Original Sample: {}, {}, {}.'.format(*variables))
            ax1.scatter(original_data[variables[0]], original_data[variables[1]], original_data[variables[2]], c=original_data[variables[2]], cmap="Spectral")
            ax1.set_xlabel(variables[0])
            ax1.set_ylabel(variables[1])
            ax1.set_zlabel(variables[2])

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title('Synthetic Sample: {}, {}, {}.'.format(*variables))
            ax2.scatter(syn_data[variables[0]], syn_data[variables[1]], syn_data[variables[2]], c=syn_data[variables[2]] ,cmap="Spectral")
            ax2.set_xlabel(variables[0])
            ax2.set_ylabel(variables[1])
            ax2.set_zlabel(variables[2])
            plt.show()
        else:
            print("variable list's length must be 1, 2 or 3. Cannot plot more than 3 variables.")

def _aligned_metric_frames(original_data, syn_data):
    original = validate_numeric_dataframe(original_data, name="original_data")
    synthetic = validate_numeric_dataframe(syn_data, name="syn_data")
    if list(original.columns) != list(synthetic.columns):
        missing_in_synthetic = [column for column in original.columns if column not in synthetic.columns]
        extra_in_synthetic = [column for column in synthetic.columns if column not in original.columns]
        raise ValueError(
            "original_data and syn_data must have the same columns in the same order. "
            f"Missing in syn_data: {missing_in_synthetic}; extra in syn_data: {extra_in_synthetic}."
        )
    return original, synthetic

def kolmogorov_distances(original_data, syn_data):
    """Return per-column Kolmogorov-Smirnov distances."""
    original, synthetic = _aligned_metric_frames(original_data, syn_data)
    distances = {
        column: ks_2samp(
            original[column].dropna().to_numpy(),
            synthetic[column].dropna().to_numpy()
        ).statistic
        for column in original.columns
    }
    return Series(distances, name="ks_statistic")

def compareStats(original_data, syn_data):
    """Compare univariate statistics for original and synthetic data.

    The returned DataFrame includes mean, standard deviation, min/max,
    Kolmogorov-Smirnov statistic and Wasserstein distance for each column.
    """
    original, synthetic = _aligned_metric_frames(original_data, syn_data)
    rows = []
    for column in original.columns:
        original_col = original[column].dropna().to_numpy()
        synthetic_col = synthetic[column].dropna().to_numpy()
        ks_result = ks_2samp(original_col, synthetic_col)
        rows.append({
            "variable": column,
            "original_mean": float(np.mean(original_col)),
            "synthetic_mean": float(np.mean(synthetic_col)),
            "mean_difference": float(np.mean(synthetic_col) - np.mean(original_col)),
            "original_std": float(np.std(original_col, ddof=1)) if original_col.size > 1 else 0.0,
            "synthetic_std": float(np.std(synthetic_col, ddof=1)) if synthetic_col.size > 1 else 0.0,
            "std_difference": (
                float(np.std(synthetic_col, ddof=1) - np.std(original_col, ddof=1))
                if min(original_col.size, synthetic_col.size) > 1
                else 0.0
            ),
            "original_min": float(np.min(original_col)),
            "synthetic_min": float(np.min(synthetic_col)),
            "original_max": float(np.max(original_col)),
            "synthetic_max": float(np.max(synthetic_col)),
            "ks_statistic": float(ks_result.statistic),
            "ks_pvalue": float(ks_result.pvalue),
            "wasserstein_distance": float(wasserstein_distance(original_col, synthetic_col)),
        })
    return DataFrame(rows).set_index("variable")

def quality_report(original_data, syn_data):
    """Return per-variable and overall utility metrics."""
    original, synthetic = _aligned_metric_frames(original_data, syn_data)
    per_variable = compareStats(original, synthetic)

    if original.shape[1] > 1:
        original_corr = original.corr().fillna(0.0)
        synthetic_corr = synthetic.corr().fillna(0.0)
        corr_diff = (original_corr - synthetic_corr).abs().to_numpy()
        upper_triangle = np.triu_indices_from(corr_diff, k=1)
        mean_abs_corr_diff = float(corr_diff[upper_triangle].mean()) if upper_triangle[0].size else 0.0
        max_abs_corr_diff = float(corr_diff[upper_triangle].max()) if upper_triangle[0].size else 0.0
    else:
        mean_abs_corr_diff = 0.0
        max_abs_corr_diff = 0.0

    overall = Series({
        "mean_ks_statistic": float(per_variable["ks_statistic"].mean()),
        "max_ks_statistic": float(per_variable["ks_statistic"].max()),
        "mean_wasserstein_distance": float(per_variable["wasserstein_distance"].mean()),
        "mean_abs_correlation_difference": mean_abs_corr_diff,
        "max_abs_correlation_difference": max_abs_corr_diff,
    }, name="overall")

    return {"per_variable": per_variable, "overall": overall}


def sample_trivariate_xyz(size = 1000):
    x = random.beta(a=0.1, b=0.1, size=size)
    y = random.beta(a=0.1, b=0.5, size=size)
    z = random.normal(size=size) + y * 10
    return DataFrame({
        'x': x,
        'y': y,
        'z': z})

def sample_circulars_xy(size):
    r = random.choice([8, 20], size = size)
    angles = random.uniform(0, 2 * pi, size)
    x = r * cos(angles) + random.randn(size)
    y = 0.5 * x -0.05 * square(x) + r * sin(angles) + random.randn(size)
    return(DataFrame({"x": x,
    "y": y}))


def new_cluster_sizes(c, n):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    counts = Series(c).astype(float)
    if counts.empty:
        raise ValueError("c must contain at least one cluster.")
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("c must contain non-negative counts with a positive total.")

    raw = counts / counts.sum() * n
    resized = np.floor(raw).astype(int)
    remainder = int(n - resized.sum())

    if remainder > 0:
        order = (raw - resized).sort_values(ascending=False).index
        for index in order[:remainder]:
            resized.loc[index] += 1
    elif remainder < 0:
        order = resized.sort_values(ascending=False).index
        to_remove = -remainder
        for index in order:
            if to_remove == 0:
                break
            removable = min(to_remove, int(resized.loc[index]))
            resized.loc[index] -= removable
            to_remove -= removable

    return resized.astype(int)
    


def compute_k_distances(data, K=5):
    """
    For each observation in data, compute the sum of Euclidean distances to its K nearest neighbors (excluding itself).
    :param data: pandas.DataFrame or numpy.ndarray
    :param K: int, number of neighbors
    :return: numpy.ndarray of shape (n_samples,)
    """
    if K is None:
        K = 5
    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer.")

    if hasattr(data, 'values'):
        X = data.values
    else:
        X = np.asarray(data)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.isfinite(X).all():
        raise ValueError("data must contain finite numeric values.")

    n = X.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.zeros(1)

    neighbor_count = min(K, n - 1)
    nbrs = NearestNeighbors(n_neighbors=neighbor_count + 1, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    # Exclude the first column (distance to self)
    k_distances = distances[:, 1:].sum(axis=1)
    return k_distances
