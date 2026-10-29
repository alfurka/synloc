Methods
=======

These convenience classes implement the covariance-based Local Resampler
methods used in the README examples.

.. autoclass:: synloc.LocalCov
    :members:
    :undoc-members:


.. autoclass:: synloc.clusterCov
    :members:
    :undoc-members:

.. autoclass:: synloc.ClusterCovSampler
    :members:

``clusterCov.fit_model()`` is an opt-in parameter-only workflow. It leaves legacy
``fit()`` clipping and diagnostics unchanged. Export requires an explicit research
review acknowledgement and never establishes privacy approval; see
:doc:`../usage` for the NPZ schema and contributor-review limitations.
