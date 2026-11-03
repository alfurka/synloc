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

For reproducible fitting, pass an integer seed to the constructor:

.. code-block:: python

   resampler = s.LocalCov(data, K=30, n_jobs=2, random_state=42)
   synthetic = resampler.fit(sample_size=500)

The seed controls anchor selection and NumPy/SciPy random draws, independently
of the worker count in the same numerical environment. Seeded parallel fitting
uses joblib's ``loky`` processes even under a threading backend override; this
prevents concurrent callbacks from racing on NumPy's global RNG. The caller's
NumPy RNG state is restored. Custom ``kNNResampler`` callbacks using their own
random generators are not controlled by this seed. ``random_state=None`` keeps
legacy RNG behaviour; ``np.random.seed`` alone does not make unseeded parallel
workers reproducible.

Count arguments accept Python and NumPy integers, but not booleans or fractions.
If a new fitting attempt fails during sampling or diagnostics, ``fitted`` is
false and previous synthetic outputs and metrics are cleared. Median imputation
continues to update the resampler's internal reference data, not the caller's
DataFrame.

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

Distance and normalization
--------------------------

Both ``LocalCov`` and ``clusterCov`` use Euclidean distance by default. When
``normalize=True`` (the default), each nonconstant variable is divided by its
sample standard deviation before selecting neighbours or clusters. Set
``normalize=False`` to skip scaling. Constant variables are left unscaled.
Cluster minimum-size reassignment uses the same scaled distance as the initial
clustering. kNN callers may specify a different neighbour metric through
``Args_NearestNeighbors``. Local distributions are fitted in original units.

.. code-block:: python

   local = s.LocalCov(data, K=30, normalize=True)
   clustered = s.clusterCov(data, n_clusters=20, normalize=True)

Parameter-only LocalCov models
------------------------------

``LocalCov.prepare_export`` selects anchors before estimating their local
Gaussian distributions, while every neighbourhood is found in the full input
data. The returned fixed proposal can be reviewed and exported without
reselecting components:

.. code-block:: python

   local = s.LocalCov(data, K=100, normalize=True)
   proposal = local.prepare_export(n_components=50, random_state=42,
                                   minimum_contributors=100)
   report = proposal.clearance_report()  # Internal; not release approval.
   direct = proposal.sample(500, random_state=123)
   proposal.export_model("local-model.npz", reviewed=True)
   sampler = s.ClusterCovSampler.load("local-model.npz")
   assert direct.equals(sampler.sample(500, random_state=123))

``n_components=None`` selects every row; a smaller positive integer selects
distinct anchors uniformly without replacement. A count larger than the input
is rejected. ``prepare_export(random_state=...)`` controls selection, not
synthetic generation, and is separate from the constructor's fitting seed.
``K`` controls the number of rows in each neighbourhood, capped by the source
row count. If supplied, ``minimum_contributors`` is a required row-count
threshold and failed components block export; otherwise the check is
unassessed. Counts mean people only if each source row is a distinct person.
Dominance diagnostics, constants and rank describe the selected neighbourhoods;
the report is confidential and is not included in the model file.

The selected components have equal weights and are exported in the same
versioned parameter-only format as ``clusterCov``. Means and covariances are in
original units, even if selection uses scaled features. The external sampler
does not clip to confidential minima/maxima, regardless of legacy ``clipping``.
This can generate out-of-range values. The legacy ``fit()`` method remains available.
With ``K=1`` a component mean can reproduce an original row exactly.
All-component exports store dense covariance matrices: memory grows as
``n_components * n_columns**2`` before sampling factors and temporary arrays.
Use an explicit smaller component count when the full mixture is too large.

Fewer components do not establish privacy or regulatory clearance. Neighbourhoods
overlap; means, covariances and multiple exports may jointly reveal sensitive
information. ``reviewed=True`` acknowledges unresolved review and is not
approval. A subset also changes the fitted mixture and may reduce utility.

Parameter-only clusterCov models
----------------------------------------

The opt-in ``clusterCov`` workflow fits once and samples without retaining
access to source records in the exported sampler. Its existing ``fit()`` method
remains available.

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
``means_``, ``covariances_``, ``weights_`` and ``columns`` expose read-only fitted
parameters on the model, LocalCov proposal and independent sampler. Refit rather
than mutating them, so sampling, export and review evidence cannot diverge.
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
covariance symmetry and positive semidefiniteness. Covariance validation and
factorization use variance-scaled units so large variances cannot hide invalid
smaller-scale blocks. Zero-variance dimensions require exactly zero covariance
rows and columns. These are mechanical checks, not privacy checks.

Version 1 files are unchanged. The factorization correction can change the
exact seeded draws relative to earlier local builds, while direct and reloaded
draws remain identical within this build and numerical environment.

Privacy review and contributor diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
