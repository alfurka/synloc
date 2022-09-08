# synloc: A Local Resampler Algorithm to Create Synthetic Data

`synloc` is an algorithm to sequentially and locally estimate distributions to create synthetic values from a sample. The proposed methodology can be combined with parametric and nonparametric distributions. 

# Installation

`synloc` can be installed through [PyPI](https://pypi.org/):

```
pip install synloc
```
(not done yet)
# Examples

Assume that we have a sample with three variables with the following distributions:

$$
\begin{aligned}
x &\sim Beta(0.1,\,0.1)\\
y &\sim Beta(0.1,\, 0.5)\\
z &\sim 10 * y + Normal(0,\,1)
\end{aligned}
$$

The distribution can be generated by `tools` module in `synloc`:


```python
from synloc.tools import sample_trivariate_xyz
data = sample_trivariate_xyz() # Generates a sample with size 1000 by default. 
```

## Creating synthetic values with Multivariate Normal Distribution

Initializing the resampler:


```python
from synloc import LocalCov
resampler = LocalCov(data = data, K = 30)
```

**Subsample** size is defined as `K=30`. Now, we locally estimate the multivariate normal distribution and from each estimated distributions we draw "synthetic values."


```python
syn_data = resampler.fit() 
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:01<00:00, 784.56it/s]
    

The default sample size is the size of the original sample (i.e., 1000). It can be changed while fitting the distributions:

```python
syn_data = resampler.fit(100) # a sample with size 100 created.
```


`syn_data` is a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) where all variables are synthesized. 

Comparing the original sample using a 3-D Scatter:


```python
resampler.comparePlots(['x','y','z'])
```


    
![png](README_files/README_7_0.png)
    


## Creating synthetic values with Gaussian Copula

Initializing the resampler:


```python
from synloc import LocalGaussianCopula
resampler = LocalGaussianCopula(data = data, K = 30)
```

We locally estimate the `Gaussian Copula` and from each estimated distributions we draw "synthetic values."


```python
syn_data_copula = resampler.fit() 
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:03<00:00, 300.94it/s]
    

Comparing the original sample using a 3-D Scatter:


```python
resampler.comparePlots(['x','y','z'])
```

![png](README_files/README_13_0.png)

Even though the distribution of the original sample substantially differs from multivariate normal, locally estimated distributions can replicate the original distribution quite accurately. 

# Customized Models

`kNNResampler` class can be used to utilize estimate and resample from any distributions. 

## functional Principal Component Analysis (fPCA)

For example, if the original sample is high-dimensional data, the user can reduce the dimension with **fPCA**. It is possible to implement it with [FPCADataGenerator](https://dmey.github.io/synthia/fpca.html) function in [Synthia](https://github.com/dmey/synthia) package:


```python
from synloc import kNNResampler
from synloc.tools import stochastic_rounder
from synthia import FPCADataGenerator

class LocalFPCA(kNNResampler):
    def __init__(self, data, K = 30, normalize = True, clipping = True, Args_NearestNeighbors = {}):
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method = self.method)
    def method(self, data):
        generator = FPCADataGenerator()
        generator.fit(data, n_fpca_components=2)
        return generator.generate(1)[0]
```

Using `kNNResampler` as a parent class, we created `LocalFPCA`. The key component is defining the method for `kNNSampler`. 

```python
super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method = self.method)
```

In the `LocalFPCA` class we define the `self.method`:

```python
def method(self, data):
    generator = FPCADataGenerator()
    generator.fit(data, n_fpca_components=2) # reducing dimension to 2
    return generator.generate(1)[0]
```


### Example


```python
resampler = LocalFPCA(data = data)
resampler.fit()
resampler.comparePlots(['x','y','z'])
```

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:02<00:00, 426.86it/s]
    


    
![png](README_files/README_17_1.png)
    


## Problem with Discrete Variables

So far, we assumed that original data set contains only continuous variables. It is possible to address this problem by rounding these variables (stochastically or deterministically), or we can use some multivariate distributions that can handle the mixed type variables. Then, we need to define a new a subclass of `kNNSampler`. 

One solution is it use [mixedvines](https://github.com/asnelt/mixedvines) package. It allows to specify the discrete type variables. 


```python
from synloc import kNNResampler
from mixedvines.mixedvine import MixedVine # pip install mixedvines


class LocalMixedVine(kNNResampler):
    def __init__(self, data, cont_cols, K = 30, normalize = True, clipping = True, Args_NearestNeighbors = {}):
        super().__init__(data, K, normalize, clipping, Args_NearestNeighbors, method = self.method)
        self.cont_cols = cont_cols
    
    def method(self, data):
        generator = MixedVine.fit(data.values, self.cont_cols)
        return generator.rvs(1)[0]
```

`LocalMixedVine` takes the argument `cont_cols` which is a boolean list. `True` if it is a continuous variable, `False` if it is discrete. Further, specification can be done following the [documentation](https://mixedvines.readthedocs.io/en/latest/). 

### Example


```python
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning) 

data2 = pd.read_stata('../extract.dta')
data2 = data2[['age','educ', 'annwage']]
data2 = data2[~data2.isna().any(1)]
data2.age = data2.age.astype('int')
data2.educ = data2.educ.astype('int')
data2 = data2.sample(1000)

resampler = LocalMixedVine(data = data2, K = 50, cont_cols = [False, False, True])
resampler.fit()
resampler.comparePlots(['age','educ', 'annwage'])
```

    100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:10<00:00,  7.66it/s]
    


    
![png](README_files/README_21_1.png)
    



```python
# Original sample looks like
data2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>educ</th>
      <th>annwage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14862</th>
      <td>30</td>
      <td>12</td>
      <td>12500.0</td>
    </tr>
    <tr>
      <th>3049</th>
      <td>28</td>
      <td>12</td>
      <td>4000.0</td>
    </tr>
    <tr>
      <th>10032</th>
      <td>34</td>
      <td>9</td>
      <td>17000.0</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>31</td>
      <td>13</td>
      <td>12000.0</td>
    </tr>
    <tr>
      <th>11450</th>
      <td>31</td>
      <td>16</td>
      <td>16000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#synthetic sample looks like
resampler.synthetic.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>educ</th>
      <th>annwage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>12</td>
      <td>14600.214734</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28</td>
      <td>11</td>
      <td>10401.463701</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>11</td>
      <td>15603.261378</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31</td>
      <td>12</td>
      <td>7839.435660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>14</td>
      <td>13485.131071</td>
    </tr>
  </tbody>
</table>
</div>


