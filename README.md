<div align="center">

# synloc: An Algorithm to Create Synthetic Tabular Data

<img src="https://raw.githubusercontent.com/alfurka/synloc/main/assets/logo_white_bc.png" alt = 'synloc'>

[Overview](#overview) | [Installation](#installation) | [A Quick Example](#a-quick-example) | [Documentation](https://alfurka.github.io/synloc/) | [How to cite?](#how-to-cite)

[![PyPI](https://img.shields.io/pypi/v/synloc)](https://pypi.org/project/synloc) [![Downloads](https://static.pepy.tech/badge/synloc)](https://pepy.tech/project/synloc)

</div>

## Overview

`synloc` is an algorithm to sequentially and locally estimate distributions to create synthetic versions of a tabular data. The proposed methodology can be combined with parametric and nonparametric distributions. 

## Installation

`synloc` can be installed through [PyPI](https://pypi.org/):

```
pip install synloc
```

## A Quick Example

Assume that we have a sample with three variables with the following distributions:

$$x \sim Beta(0.1,\,0.1)$$
$$y \sim Beta(0.1,\, 0.5)$$
$$z \sim 10 y + Normal(0,\,1)$$

The distribution can be generated by `tools` module in `synloc`:


```python
from synloc.tools import sample_trivariate_xyz
data = sample_trivariate_xyz() # Generates a sample with size 1000 by default. 
```

Initializing the resampler:


```python
from synloc import LocalCov
resampler = LocalCov(data = data, K = 30)
```

**Subsample** size is defined as `K=30`. Now, we locally estimate the multivariate normal distribution and from each estimated distributions we draw "synthetic values."


```python
syn_data = resampler.fit() 
```

    100%|██████████| 1000/1000 [00:01<00:00, 687.53it/s]
    

`syn_data` is a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) where all variables are synthesized. Comparing the original sample using a 3-D Scatter:


```python
resampler.comparePlots(['x','y','z'])
```    
![](https://raw.githubusercontent.com/alfurka/synloc/main/assets/README_7_0.png)

## How to cite?

If you use `synloc` in your research, please cite the following paper:

```bibtex
@misc{kalay2025generatingsyntheticdatalocally,
      title={Generating Synthetic Data with Locally Estimated Distributions for Disclosure Control}, 
      author={Ali Furkan Kalay},
      year={2025},
      eprint={2210.00884},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
      url={https://arxiv.org/abs/2210.00884}, 
}
```
