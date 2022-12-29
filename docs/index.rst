Welcome to synloc's documentation!
==================================

.. image:: https://raw.githubusercontent.com/alfurka/synloc/main/assets/logo.png
   :alt: synloc
   :align: center

``synloc`` is an algorithm to sequentially and locally estimate distributions to create synthetic versions of a tabular data. The proposed methodology can be combined with parametric and nonparametric distributions. 

Installation
##################


``synloc`` can be installed through pip:

.. code-block:: python
   :linenos:

   pip install synloc

A Quick Example
#################

.. code-block:: python
   :linenos:

   from synloc.tools import sample_trivariate_xyz
   from synloc import LocalCov
   data = sample_trivariate_xyz() # Generates a sample with size 1000 by default. 
   resampler = LocalCov(data = data, K = 30)
   syn_data = resampler.fit() # data is saved to `syn_data`
   resampler.comparePlots(['x','y','z'])

.. image:: https://raw.githubusercontent.com/alfurka/synloc/main/README_files/README_7_0.png
   :alt: synloc
   :align: center

Documentation
##################

* :doc:`Notebooks/nearest_neighbor`
* :doc:`Notebooks/clustering`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation
   
   Notebooks/nearest_neighbor
   Notebooks/clustering

API
##################

* :doc:`api/resamplers`
* :doc:`api/methods`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API
   
   api/resamplers
   api/methods

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
