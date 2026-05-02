# Replication Files

This folder contains the replication materials for:

**Kalay, A. F. (2026).** *Generating Synthetic Data With Locally Estimated Distributions for Disclosure Control.* Australian & New Zealand Journal of Statistics. [https://doi.org/10.1111/anzs.70032](https://doi.org/10.1111/anzs.70032)

The notebook is intended to reproduce the tables, simulations, and figures reported in the paper. Because several comparison methods changed their APIs after the analysis was completed, the safest route is to use the package versions printed in the notebook output.

---

## Main Replication File

**[Replication Notebook.ipynb](Replication%20Notebook.ipynb)**

This Jupyter notebook is the primary replication file for the paper. It contains the simulations, analyses, and figures. Some intermediate CSV files are retained because parts of the comparison workflow involve nested randomness that could not be fully controlled by a single random seed.

---

## Dependencies

### Python Packages

```python
from tqdm.autonotebook import tqdm as notebook_tqdm
import synloc as s
import matplotlib.pyplot as plt
import seaborn as sns  # for KDE plots
import pandas as pd
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sdv.metrics.tabular import KSComplement, CorrelationSimilarity
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata
import numpy as np
np.random.seed(42)
```

Specific package versions are printed within the notebook.

> [!WARNING]  
> **SDV Package Instability:**  
> The `SDV` (Synthetic Data Vault) package has undergone at least three major revisions to its API. We **strongly recommend** using the exact version specified in the notebook to ensure reproducibility.

### Modified `anonymeter` Package

The [`anonymeter/`](anonymeter/) folder contains a **modified fork** of the [statice/anonymeter](https://github.com/statice/anonymeter) package. This modification was necessary to resolve a dependency conflict between `synloc` and `anonymeter` related to Python version requirements.

This package is used for **singling out risk computation**. Note that this is not a central metric to the paper, but is included for completeness.

> [!NOTE]  
> You may not be able to install all required packages via `pip` due to this conflict. If package resolution fails, use the modified `anonymeter` version provided here alongside the exact package versions printed in the notebook.

### R (for `synthpop`)

**[synthpop.R](synthpop.R)**

This script uses the R `synthpop` package for comparison purposes. It is not a major component of the replication and should not have dependency issues.

```r
library(synthpop)
```

---

## Intermediate Datasets

The following CSV files are **intermediate datasets** generated during simulations. These are preserved so that paper outputs can be checked without rerunning every stochastic comparison model.

| File | Description |
|------|-------------|
| `california.csv` | California housing dataset |
| `california_synthpop.csv` | Synthetic version via `synthpop` |
| `circulars.csv` | Circular distribution data |
| `circulars_synthpop.csv` | Synthetic version via `synthpop` |
| `trivariate.csv` | Trivariate distribution data |
| `trivariate_synthpop.csv` | Synthetic version via `synthpop` |
| `discrete_data.csv` | Discrete data |
| `discrete_data_synthpop.csv` | Synthetic version via `synthpop` |

---

## Notes

- Random seed is set to `42`
- The notebook outputs exact package versions for full reproducibility
- If you encounter issues, please refer to the exact versions listed in the notebook output
