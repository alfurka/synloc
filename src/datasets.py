import pandas as pd
import numpy as np
from scipy import stats

def sample_trivariate_xyz(size = 1000):
    x = stats.beta.rvs(a=0.1, b=0.1, size=size)
    y = stats.beta.rvs(a=0.1, b=0.5, size=size)
    return pd.DataFrame({
        'x': x,
        'y': y,
        'z': np.random.normal(size=size) + y * 10})