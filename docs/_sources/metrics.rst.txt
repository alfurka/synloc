Quality metrics
===============

Version 1.0 adds utility diagnostics that can be used immediately after fitting.
These metrics are descriptive checks, not formal privacy guarantees.

Per-variable statistics
-----------------------

Use ``compareStats`` to compare each original and synthetic column:

.. code-block:: python

   stats = resampler.compareStats()
   print(stats[["original_mean", "synthetic_mean", "ks_statistic"]])

The returned ``DataFrame`` includes:

* original and synthetic means
* mean difference
* original and synthetic standard deviations
* standard-deviation difference
* original and synthetic minimum and maximum
* Kolmogorov-Smirnov statistic and p-value
* Wasserstein distance

Overall report
--------------

Use ``qualityReport`` on a fitted resampler:

.. code-block:: python

   report = resampler.qualityReport()
   print(report["overall"])

The report contains:

* ``per_variable``: the same table returned by ``compareStats``
* ``overall``: mean and maximum Kolmogorov-Smirnov statistic, mean
  Wasserstein distance, and correlation-difference summaries

Function API
------------

The same metrics are available as functions:

.. code-block:: python

   from synloc import compareStats, quality_report, kolmogorov_distances

   stats = compareStats(original_data, synthetic_data)
   ks = kolmogorov_distances(original_data, synthetic_data)
   report = quality_report(original_data, synthetic_data)

Interpreting metrics
--------------------

Lower values generally indicate closer agreement between original and synthetic
data. The right threshold depends on the application, sample size, and privacy
requirements. Use these diagnostics alongside domain checks and disclosure-risk
assessment when synthetic data will be shared.
