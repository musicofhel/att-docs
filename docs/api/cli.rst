att.cli
=======

.. automodule:: att.cli
   :members:
   :undoc-members:

.. automodule:: att.cli.main
   :members:

Command-Line Usage
------------------

.. code-block:: text

   att benchmark run --config experiment.yaml --output results.csv [--plot sweep.png]

Run a coupling benchmark sweep defined in a YAML config file.  Results are
written to a CSV with columns ``coupling``, ``method``, ``score``, and
``score_normalized``.
