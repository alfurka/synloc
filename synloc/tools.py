import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def fill_na_with_median(dataframe):
    dataframe = dataframe.fillna(dataframe.median(0))
    dataframe = dataframe.reset_index(drop = True)
    print('Missing values are filled with variable medians.')
    return dataframe

def stochastic_integer(x):
    lower_int = np.floor(x)
    upper_int = lower_int + 1
    return (np.random.choice([lower_int, upper_int], p=[upper_int - x, x - lower_int]))

stochastic_rounder = np.vectorize(stochastic_integer)

def stochastic_up_or_down(dataframe, p):
    up_or_down = 2 * (np.random.binomial(1, 0.5, dataframe.shape) - 0.5)
    outcomes = np.random.binomial(1, p, dataframe.shape)
    new_dataframe = dataframe + up_or_down * outcomes
    return new_dataframe.clip(lower=dataframe.min(0), upper=dataframe.max(0), axis = 1)

def compareplots(original_data, syn_data, variable, fig_size = (10,8)):
    if (type(variable) == str) | (len(variable) == 1):
        assert variable in original_data.columns
        if fig_size is None:
            fig_size = (10, 8)
        plt.figure(figsize = fig_size)
        plt.title('Original and Synthetic values of variable `{}`'.format(variable))
        plt.hist(original_data[variable], alpha = 0.5, label = 'Original Sample')
        plt.hist(syn_data[variable], alpha = 0.5, label = 'Synthetic Sample')
        plt.legend(loc='upper right')
        plt.show()
    else:
        if len(variable) == 2:
            if fig_size is None:
                fig_size = (15,7)          
            fig = plt.figure(figsize = fig_size)
            ax1 = fig.add_subplot(121)
            ax1.set_title('Original sample: `{}` and `{}`'.format(*variable))
            h1 = ax1.hist2d(original_data[variable[0]] , original_data[variable[1]], cmap = 'jet')
            plt.colorbar(h1[3])

            ax2 = fig.add_subplot(122)
            ax2.set_title('Synthetic sample: `{}` and `{}`'.format(*variable))
            h2 = ax2.hist2d(syn_data[variable[0]] , syn_data[variable[1]], cmap = 'jet')
            plt.colorbar(h2[3])
            plt.show()
        
        elif len(variable) == 3:
            if fig_size is None:
                fig_size = (15,7) 
            fig = plt.figure(figsize = fig_size)
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Original Sample: {}, {}, {}.'.format(*variable))
            ax1.scatter(original_data[variable[0]], original_data[variable[1]], original_data[variable[2]], c=original_data[variable[2]], cmap="Spectral")
            ax1.set_xlabel(variable[0])
            ax1.set_ylabel(variable[1])
            ax1.set_zlabel(variable[2])

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title('Synthetic Sample: {}, {}, {}.'.format(*variable))
            ax2.scatter(syn_data[variable[0]], syn_data[variable[1]], syn_data[variable[2]], c=syn_data[variable[2]] ,cmap="Spectral")
            ax2.set_xlabel(variable[0])
            ax2.set_ylabel(variable[1])
            ax2.set_zlabel(variable[2])
            plt.show()
        else:
            print("variable list's length must be 1, 2 or 3. Cannot plot more than 3 variables.")
def compareStats():
    pass


def sample_trivariate_xyz(size = 1000):
    x = stats.beta.rvs(a=0.1, b=0.1, size=size)
    y = stats.beta.rvs(a=0.1, b=0.5, size=size)
    return pd.DataFrame({
        'x': x,
        'y': y,
        'z': np.random.normal(size=size) + y * 10})