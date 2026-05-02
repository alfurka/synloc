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
