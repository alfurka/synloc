synloc documentation
====================

``synloc`` implements the Local Resampler algorithm for creating synthetic
numeric tabular data from locally estimated distributions.

The public API is intentionally small:

* ``LocalCov`` uses nearest-neighbor neighborhoods.
* ``clusterCov`` uses KMeans clusters.
* ``compareStats`` and ``quality_report`` summarize utility diagnostics.

Version 1.0 keeps the existing constructor and method names while adding
clearer input checks, quality metrics, tests, and release documentation.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   input_requirements
   usage
   metrics
   replication
   release_notes

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api

Citation
--------

If you use ``synloc`` in research, please cite:

Kalay, A. F. (2026). *Generating Synthetic Data With Locally Estimated
Distributions for Disclosure Control*. Australian & New Zealand Journal of
Statistics. https://doi.org/10.1111/anzs.70032

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
