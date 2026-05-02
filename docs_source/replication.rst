Replication materials
=====================

The ``replication`` folder contains the notebook, scripts, and intermediate
datasets used for the paper:

Kalay, A. F. (2026). *Generating Synthetic Data With Locally Estimated
Distributions for Disclosure Control*. Australian & New Zealand Journal of
Statistics. https://doi.org/10.1111/anzs.70032

The main file is ``replication/Replication Notebook.ipynb``.

Reproducibility notes
---------------------

Some comparison packages, especially SDV, changed their APIs after the paper
analysis was completed. The replication notebook prints exact package versions;
use those versions when reproducing paper outputs.

The folder also includes a modified copy of ``anonymeter``. It is used for the
singling-out risk calculation and is included to avoid a dependency conflict
with the historical replication environment.

Intermediate CSV files are retained because parts of the comparison workflow
involve nested randomness. Keeping these files makes it possible to check paper
outputs without rerunning every stochastic comparison model.
