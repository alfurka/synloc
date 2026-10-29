Usage
=====

Nearest-neighbor resampling
---------------------------

``LocalCov`` estimates a local covariance matrix from each observation's
nearest-neighbor neighborhood and draws one synthetic value from the estimated
multivariate normal distribution.

.. code-block:: python

   import synloc as s

   data = s.sample_trivariate_xyz(1000)
   resampler = s.LocalCov(data=data, K=30, n_jobs=1)
   synthetic = resampler.fit()

``synthetic`` is a ``pandas.DataFrame`` with the same columns as ``data``.
If a variable is constant inside a local neighborhood, that variable is copied
exactly for the synthetic value drawn from that neighborhood.

Use ``sample_size`` to request a different synthetic sample size:

.. code-block:: python

   synthetic = resampler.fit(sample_size=500)

Cluster resampling
------------------

``clusterCov`` clusters the data with KMeans, estimates a covariance matrix
inside each cluster, and draws synthetic observations cluster by cluster.
If a variable is constant inside a cluster, that variable is copied exactly for
synthetic rows generated from that cluster.

.. code-block:: python

   import synloc as s

   data = s.sample_circulars_xy(1000)
   resampler = s.clusterCov(data=data, n_clusters=20, size_min=8)
   synthetic = resampler.fit()

Parameter-only clusterCov models
-------------------------------

The opt-in model workflow fits once and samples without retaining access to
source records in the exported sampler. It is available only for ``clusterCov``;
``LocalCov`` and legacy ``fit()`` behavior are unchanged.

.. code-block:: python

   model = s.clusterCov(data, n_clusters=20, size_min=8).fit_model()
   report = model.clearance_report()  # Internal: contains source-derived counts.
   direct = model.sample(500, random_state=42)
   # Research export acknowledgement, NOT privacy or regulatory approval:
   model.export_model("cluster-model.npz", reviewed=True)
   sampler = s.ClusterCovSampler.load("cluster-model.npz")
   restored = sampler.sample(500, random_state=42)
   assert direct.equals(restored)

``fit_model()`` returns the model, not synthetic rows, and does not run utility
diagnostics. ``sample(n, random_state=None)`` returns a DataFrame in the original
column order; nonnegative integer sizes (including zero) are accepted. Integer
seeds give exact direct/load equality in the same numerical environment.
``means_``, ``covariances_``, ``weights_`` and ``columns`` expose fitted parameters.
Use legacy ``fit()`` for the plotting and quality-report workflow.

Normalization is used only to determine clusters. Means and covariances are
estimated in original units after median imputation; mixture weights are final
source cluster counts divided by the source row count, never requested synthetic
sample allocations. Locally constant columns are exact. Varying covariance blocks
use the existing regularization. There is no clipping to confidential source
minima or maxima, even if the constructor's legacy ``clipping`` option is true.
Consequently a Gaussian draw can be negative for an otherwise nonnegative
variable; support constraints and transformed models are not supplied by this API.

Version 1 NPZ files contain exactly ``version`` (integer scalar), ``column_names``
(unique nonempty Unicode names), ``weights`` (positive normalized mixture weights),
``means`` (components by columns) and ``covariances`` (components by columns by
columns). No records, labels, explicit counts, source bounds, normalization factors,
or internal clearance reports are serialized. Loading disables pickle and checks
the exact schema, version, dimensions, finiteness, names, weight normalization,
covariance symmetry and positive semidefiniteness. These are mechanical checks,
not privacy checks.

Privacy review and contributor diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``clearance_report()`` is an internal JSON-serializable report, not a release
certificate. ``mechanical_validity`` is separate from ``privacy_approved``, which
is always false. ``required_contributor_check`` checks actual final cluster row
counts against ``size_min``; it is unassessed when no minimum is configured.
**One row must represent one person** for the row count to be interpreted as
person contributors. Repeated persons and household dependencies are not checked.
No clusters are silently dropped; failed required contributor checks block export
even with ``reviewed=True``.

Per-component diagnostics include counts, empirical centered-data rank, constant
columns, and nonnegative-variable dominance. The largest contributor must account
for at most 50 percent and the largest two for at most 67 percent to pass the
respective 1/50 and 2/67 diagnostics. Zero totals are unassessed; negative-valued
variables are not applicable. These variable-level diagnostics do not assess
covariance cross-products or the joint release and are not additional automatic
export gates.

Every export requires the explicit keyword ``reviewed=True``. This acknowledges
unresolved privacy review for research use; it is not an assertion that an
authority has approved release. Weights can reveal counts when the total source
size is known. Means, covariance entries, constant values and repeated samples
can disclose fitted parameters. Household relationships, repeated persons,
covariance disclosure and count leakage require separate review. Keep the
source-derived internal report separate from any released artifact. Neither a
parameter-only format nor synthetic sampling provides a privacy guarantee.

Visualization
-------------

After fitting, compare one, two, or three variables visually:

.. code-block:: python

   resampler.comparePlots(["x", "y"])
   resampler.comparePlots(["x", "y", "z"])

The plotting helper is intended for quick diagnostics. For publication figures,
use the returned synthetic data with your preferred plotting workflow.
