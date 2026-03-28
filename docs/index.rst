Attractor Topology Toolkit
==========================

**ATT** is a Python library for topological analysis of dynamical attractors via
persistent homology on Takens embeddings.  Its core contribution is the
*joint-vs-marginal* persistence image framework: by comparing the persistent
homology of a joint delay embedding against the marginals, ATT detects
**topological binding** -- emergent structure that exists only when two systems
are coupled.

.. code-block:: python

   from att.config import set_seed
   from att.synthetic import lorenz_system
   from att.embedding import TakensEmbedder
   from att.topology import PersistenceAnalyzer

   set_seed(42)
   trajectory = lorenz_system(n_steps=5000)
   cloud = TakensEmbedder(delay=15, dimension=3).fit_transform(trajectory[:, 0])
   result = PersistenceAnalyzer(max_dim=2).fit_transform(cloud)

Key Features
------------

- **Synthetic generators** -- Lorenz, Rossler, coupled systems, switching dynamics
- **Takens & joint embedding** -- automatic delay/dimension estimation (AMI + FNN)
- **Persistent homology** -- Ripser and GUDHI backends, persistence images, Betti curves
- **Binding detection** -- joint-vs-marginal residuals with surrogate significance testing
- **Transition detection** -- sliding-window PH with CUSUM changepoint detection
- **Benchmark framework** -- compare binding score against transfer entropy, PAC, CRQA
- **EEG support** -- MNE-Python loader with auto-estimation and literature fallback params
- **Publication-quality plots** -- diagrams, barcodes, binding images, transition timelines

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
