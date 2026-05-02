Installation
============

Install from PyPI:

.. code-block:: bash

   pip install synloc

For local development, install the package with its development tools:

.. code-block:: bash

   pip install -e ".[dev]"

The package depends on ``pandas``, ``numpy``, ``scipy``, ``matplotlib``,
``scikit-learn``, ``joblib``, and ``tqdm``.

Documentation builds
--------------------

GitHub Pages is configured to publish the ``docs`` folder. Edit the files in
``docs_source`` and rebuild the HTML with:

.. code-block:: bash

   sphinx-build -b html docs_source docs

Keep ``docs/.nojekyll`` in the generated folder so GitHub Pages serves Sphinx
assets correctly.
