from numpy import random, floor, vectorize, sin, cos, pi, square, round
import matplotlib.pyplot as plt
from pandas import DataFrame

def fill_na_with_median(dataframe):
    dataframe = dataframe.fillna(dataframe.median(0))
    dataframe = dataframe.reset_index(drop = True)
    print('Missing values are filled with variable medians.')
    return dataframe

def stochastic_integer(x):
    lower_int = floor(x)
    upper_int = lower_int + 1
    return (random.choice([lower_int, upper_int], p=[upper_int - x, x - lower_int]))

stochastic_rounder = vectorize(stochastic_integer)

def stochastic_up_or_down(dataframe, p):
    up_or_down = 2 * (random.binomial(1, 0.5, dataframe.shape) - 0.5)
    outcomes = random.binomial(1, p, dataframe.shape)
    new_dataframe = dataframe + up_or_down * outcomes
    return new_dataframe.clip(lower=dataframe.min(0), upper=dataframe.max(0), axis = 1)

def compareplots(original_data, syn_data, variable, fig_size = (10,8)):
    if (type(variable) == str) | (len(variable) == 1): # Histogram
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
        if len(variable) == 2: # Scatter plot
            if fig_size is None:
                fig_size = (15,7)          
            fig = plt.figure(figsize = fig_size)
            ax1 = fig.add_subplot(121)
            ax1.set_title('Original sample: `{}` and `{}`'.format(*variable))
            ax1.scatter(original_data[variable[0]] , original_data[variable[1]])
            ax2 = fig.add_subplot(122)
            ax2.set_title('Synthetic sample: `{}` and `{}`'.format(*variable))
            ax2.scatter(syn_data[variable[0]] , syn_data[variable[1]])
            plt.show()
        
        elif len(variable) == 3: # 3d scatter plot
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
    total = c.sum()
    c = c/total * n
    c = c.round()
    c = c.astype(int)
    n_cluster = c.shape[0]
    new_total = c.sum()
    diff = new_total - n
    if diff==0:
        return c
    else:
        for i in range(abs(diff)):
            if diff > 0:
                c.iloc[i % n_cluster] -= 1
            else:
                c.iloc[i % n_cluster] += 1
        return c