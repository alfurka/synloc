Input requirements
==================

``synloc`` expects a numeric ``pandas.DataFrame``.

Before calling a resampler, prepare the data as follows:

* Encode categorical variables as numeric columns, for example with
  ``pandas.get_dummies``.
* Use one row per observation and one column per variable.
* Avoid duplicate column names.
* Avoid positive or negative infinite values.
* Do not pass columns that are entirely missing.

Missing values
--------------

Numeric missing values are filled with column medians during ``fit``. This is
intended as a convenience, not as a substitute for thoughtful preprocessing.
Columns with only missing values cannot be imputed and raise a ``ValueError``.

Dummy variables
---------------

Boolean dummy columns are accepted and converted to numeric ``0`` and ``1``
values. Other categorical columns, including strings and ``object`` columns,
raise a ``TypeError`` with the column names that need preprocessing.

Integer-like variables
----------------------

Synthetic samples are generated as continuous numeric values. If a variable
should be integer-like, call ``round_integers`` after fitting:

.. code-block:: python

   resampler.round_integers(["age", "children"], stochastic=True)
