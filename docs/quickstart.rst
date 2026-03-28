Quickstart
==========

Installation
------------

Install the core package from PyPI:

.. code-block:: bash

   pip install att-toolkit

For optional features, install extras:

.. code-block:: bash

   # EEG analysis (MNE-Python)
   pip install att-toolkit[eeg]

   # GUDHI backend for higher-dimensional homology
   pip install att-toolkit[gudhi]

   # Interactive 3D plots
   pip install att-toolkit[plotly]

   # Everything
   pip install att-toolkit[all]

   # Development (tests + linting)
   pip install att-toolkit[dev]


Example 1: Lorenz Attractor Fingerprint
----------------------------------------

This example generates a Lorenz trajectory, reconstructs the attractor from a
single scalar observable via Takens embedding, computes persistent homology, and
visualises the result.

**Step 1 -- Generate data**

.. code-block:: python

   from att.config import set_seed
   from att.synthetic import lorenz_system

   set_seed(42)
   trajectory = lorenz_system(n_steps=8000, dt=0.01)
   # trajectory.shape == (8000, 3)

   # Use the x-coordinate as our scalar time series
   x = trajectory[:, 0]

**Step 2 -- Embed**

.. code-block:: python

   from att.embedding import TakensEmbedder

   embedder = TakensEmbedder(delay="auto", dimension="auto")
   cloud = embedder.fit_transform(x)

   print(f"Estimated delay: {embedder.delay_}")
   print(f"Estimated dimension: {embedder.dimension_}")
   print(f"Point cloud shape: {cloud.shape}")

The ``"auto"`` settings use Average Mutual Information (AMI) for delay and False
Nearest Neighbors (FNN) for dimension -- the standard Takens reconstruction
pipeline.

**Step 3 -- Compute persistent homology**

.. code-block:: python

   from att.topology import PersistenceAnalyzer

   pa = PersistenceAnalyzer(max_dim=2, backend="ripser")
   result = pa.fit_transform(cloud, subsample=1500, seed=42)

   print(f"H0 features: {len(result['diagrams'][0])}")
   print(f"H1 features: {len(result['diagrams'][1])}")
   print(f"H2 features: {len(result['diagrams'][2])}")
   print(f"Persistence entropy: {result['persistence_entropy']}")

The Lorenz attractor produces a characteristic *topological fingerprint*:
one dominant H1 loop (the butterfly wings) and near-zero H2 (no enclosed voids).

**Step 4 -- Visualise**

.. code-block:: python

   from att.viz import plot_persistence_diagram, plot_barcode, plot_attractor_3d

   # Persistence diagram
   fig = plot_persistence_diagram(result["diagrams"])
   fig.savefig("lorenz_persistence.png", dpi=150)

   # Barcode
   fig = plot_barcode(result["diagrams"])
   fig.savefig("lorenz_barcode.png", dpi=150)

   # 3D attractor (requires plotly extra)
   fig = plot_attractor_3d(cloud, backend="matplotlib")
   fig.savefig("lorenz_attractor.png", dpi=150)


Example 2: Binding Detection on Coupled Lorenz
------------------------------------------------

This example demonstrates ATT's core novelty: detecting topological binding
between two coupled chaotic systems.

**Step 1 -- Generate coupled systems**

.. code-block:: python

   from att.config import set_seed
   from att.synthetic import coupled_lorenz

   set_seed(42)
   trajectory = coupled_lorenz(n_steps=8000, coupling=2.0, seed=42)
   # trajectory.shape == (8000, 6) -- two Lorenz systems, 3 vars each

   x = trajectory[:, 0]  # x-component of system 1
   y = trajectory[:, 3]  # x-component of system 2

**Step 2 -- Detect binding**

.. code-block:: python

   from att.binding import BindingDetector

   detector = BindingDetector(max_dim=1, baseline="max")
   detector.fit(x, y, subsample=1500, seed=42)

   score = detector.binding_score()
   print(f"Binding score: {score:.4f}")

   features = detector.binding_features()
   print(f"H1 excess features: {features[1]['n_excess']}")

A positive binding score means the joint embedding contains topological
structure (loops, components) that is absent from *both* marginals -- evidence
of emergent coupling structure.

**Step 3 -- Significance testing**

.. code-block:: python

   result = detector.test_significance(
       n_surrogates=99,
       method="phase_randomize",
       seed=42,
       subsample=1500,
   )

   print(f"Observed score: {result['observed_score']:.4f}")
   print(f"p-value: {result['p_value']:.4f}")
   print(f"Significant (p < 0.05): {result['significant']}")

Phase-randomized surrogates preserve the power spectrum of the second signal
while destroying nonlinear coupling.  If the observed binding score exceeds
the 95th percentile of the surrogate distribution, the coupling is
statistically significant.

**Step 4 -- Visualise**

.. code-block:: python

   from att.viz import plot_surrogate_distribution, plot_binding_image

   # Surrogate distribution with observed score
   fig = plot_surrogate_distribution(
       result["observed_score"],
       result["surrogate_scores"],
   )
   fig.savefig("binding_surrogates.png", dpi=150)

   # Residual persistence images (red = emergent topology)
   fig = plot_binding_image(detector.binding_image())
   fig.savefig("binding_image.png", dpi=150)

   # 3-panel comparison: marginal X | joint | marginal Y
   fig = detector.plot_comparison()
   fig.savefig("binding_comparison.png", dpi=150)


What next?
----------

- **Transition detection**: Use :class:`~att.transitions.TransitionDetector` to
  find topological regime changes in non-stationary time series.
- **Benchmarking**: Compare ATT's binding score against transfer entropy, PAC,
  and CRQA using :class:`~att.benchmarks.CouplingBenchmark`.
- **EEG analysis**: Load real neural data with :class:`~att.neuro.EEGLoader`
  and the auto-fallback :func:`~att.neuro.embed_channel` pipeline.
- **CLI**: Run ``att benchmark run --config experiment.yaml --output results.csv``
  from the command line.
- Full API reference: :doc:`api/index`
