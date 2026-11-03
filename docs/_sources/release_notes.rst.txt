Version 1.1 notes
=================

Version 1.1 adds optional parameter-only export for ``clusterCov`` and
``LocalCov``, plus public-schema mixed-type cluster exports. The basic
constructors, ``fit()``, ``synthetic`` and diagnostic methods remain available.
This is a GitHub source release; PyPI publication is separate. Model export does
not establish disclosure clearance.

Corrections in 1.1
------------------

* ``LocalCov`` accepts an optional trailing ``random_state``. Explicit integer
  seeds now control both kNN anchor selection and NumPy/SciPy callback draws,
  with the same results across worker counts in the same numerical environment.
  Seeded parallel calls use isolated ``loky`` processes, not a threading override.
  The caller's NumPy RNG is restored; custom callbacks with independent random
  generators are outside this guarantee. Omitting the seed retains legacy
  behaviour.
* Count arguments accept NumPy integers and reject booleans and fractions.
* Failed refits clear stale synthetic results and diagnostics instead of
  leaving an earlier fit marked as successful.
* Locally constant decimal values are preserved exactly, including without
  clipping.
* Fitted mixture parameters are read-only. Callers cannot silently change
  sampling or exported parameters independently of the fitted factors and
  internal review evidence.
* Covariance validation and factorization use variance-scaled units. Invalid
  small-scale blocks are no longer hidden by a large variance elsewhere.
  Constant dimensions require exactly zero covariance rows and columns.
* Real-valued input is required; complex source values and mixed raw scores
  are rejected before conversion can discard their imaginary parts.

The version-1 export schemas are unchanged. The corrected factorization can
change exact seeded samples relative to previous local builds. Direct and
reloaded samples still agree within the same build and numerical environment;
historical paper results are not promised to be bitwise reproducible. The
replication materials and saved research results have not been changed or rerun.

Version 1.0 notes
=================

Version 1.0 is a stability release. It keeps the existing public API while
making the package safer for PyPI users.

Highlights
----------

* Public constructors and method names are unchanged.
* Input data is checked when a resampler is created.
* Missing numeric values are still filled with medians during fitting.
* Non-numeric columns fail early with clear error messages.
* Distance diagnostics handle small samples and large ``K`` values.
* Local covariance sampling is regularized for small or singular neighborhoods.
* Variables that are constant inside a k-neighbor subsample or cluster are
  preserved exactly in the generated synthetic values.
* Unit tests cover the main resamplers, validation, metrics, and helper edge
  cases.

Recommended release checklist
-----------------------------

Before uploading to PyPI:

.. code-block:: bash

   python -m pytest
   sphinx-build -b html docs_source docs
   python -m build
   python -m twine check dist/*

Then upload:

.. code-block:: bash

   python -m twine upload dist/*
