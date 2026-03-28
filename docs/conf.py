# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root so autodoc can find the ``att`` package.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "Attractor Topology Toolkit"
copyright = "2026, ATT Contributors"
author = "ATT Contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstrings = False
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "ATT Documentation"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/musicofhel/att-docs",
    "source_branch": "master",
    "source_directory": "docs/",
}

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Mock imports for optional dependencies ----------------------------------
# Allow doc builds without heavy optional deps installed.
autodoc_mock_imports = [
    "mne",
    "openneuro",
    "gudhi",
    "plotly",
    "streamlit",
    "pyinform",
    "ripser",
    "persim",
]
