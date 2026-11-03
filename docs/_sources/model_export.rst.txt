Model export and loading
========================

Fit a reusable model, save its parameters, then generate synthetic data in
another environment without the original records. These workflows are available
in ``synloc`` 1.1. They do not require an ordinary ``fit()`` call first;
the existing ``fit()`` workflow is unchanged.

Upgrade an existing installation with:

.. code-block:: bash

   pip install --upgrade synloc

Example input
-------------

Use a numeric DataFrame with unique, nonempty string column names. Here is
standalone example data; replace it with your own numeric data when fitting:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import synloc as sl

   rng = np.random.default_rng(42)
   data = pd.DataFrame(
       rng.normal(size=(1000, 3)), columns=["x", "y", "z"]
   )

Choose either export method below. Counts are illustrative, not recommended
privacy thresholds.

Export a cluster model
----------------------

``fit_model()`` estimates the mixture without generating synthetic rows.

.. code-block:: python

   model = sl.clusterCov(
       data, n_clusters=20, size_min=20
   ).fit_model()
   # Inspect internally before export.
   diagnostics = model.disclosure_diagnostics()
   model.export_model("cluster-model.npz")

Weights reflect the final source cluster sizes, not a requested synthetic
sample size.

Export a LocalCov model
------------------------

``prepare_export()`` selects anchors and estimates their neighbourhood
distributions using the full input dataset. It returns a fixed proposal;
export that proposal, not the original resampler.

.. code-block:: python

   local = sl.LocalCov(data, K=100, normalize=True)
   proposal = local.prepare_export(n_components=50, random_state=42)
   # Inspect the selected components.
   diagnostics = proposal.disclosure_diagnostics()
   proposal.export_model("local-model.npz")

* ``K=100``: up to 100 source rows per neighbourhood.
* ``n_components=50``: export 50 mean/covariance pairs with equal weights.
  ``None`` (the default) uses every row as an anchor.
* ``random_state=42``: seed for selecting anchors, not for later sampling.

Fewer components change the fitted mixture; they do not guarantee privacy.
All-component exports can be large: covariance storage grows with
``n_components * n_columns**2``.

Load and sample elsewhere
-------------------------

Both numeric exports use **the same loader**, despite its name. With ``synloc``
installed, run this in a separate Python session with just the exported file:

.. code-block:: python

   from synloc import ClusterCovSampler

   sampler = ClusterCovSampler.load("cluster-model.npz")
   synthetic = sampler.sample(n=10000, random_state=123)

Use ``"local-model.npz"`` in the same call to load the LocalCov export.
``sample()`` returns a DataFrame in the original column order. The same integer
sampling seed reproduces direct and loaded draws in the same build and numerical
environment. The fitted ``model`` and ``proposal`` also expose ``sample()``.

What the file contains
----------------------

The version-1 NPZ stores only ``version``, ``column_names``, ``weights``,
``means`` and ``covariances``. It excludes source rows, memberships, counts,
source minima/maxima and the disclosure diagnostics.

Means and covariances are in **original units**. Normalization affects only
neighbour or cluster selection, so no scaling factors are needed when loading.
Exported sampling does not clip to source bounds; generated values may be
negative or otherwise outside the source range. Fitted parameters are read-only.

Before releasing a model
------------------------

``disclosure_diagnostics()`` returns one row per component, in the same order
as ``means_``:

* ``count``: source rows in the component.
* ``rank``: rank of the component's centred rows. A low rank means the
  component is close to a few records.
* ``constant_columns``: variables with one shared value in the component.
* ``top1_share_<column>`` and ``top2_share_<column>``: the largest one and two
  values as shares of the component total; ``NaN`` if the variable has negative
  values or a zero total.

The diagnostics are computed from the source rows, so keep them confidential;
they are not written to the exported file. They apply no thresholds and do not
establish privacy: apply your own disclosure rules before releasing a model.
Counts refer to rows, not verified unique people.

Loading validates the numerical format with pickle disabled, **not privacy**.
Means, covariances, weights, overlapping neighbourhoods and repeated exports can
still disclose information.

API
---

.. autoclass:: synloc.ClusterCovSampler
   :members:
