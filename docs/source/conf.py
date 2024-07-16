# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.append(os.path.abspath("./_ext"))
sys.path.append(os.path.abspath("."))
from _ext.github_link import make_linkcode_resolve  # noqa: E402

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SCN"
copyright = "2024, Guillermo Martin"
author = "Guillermo Martin"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
]

autodoc_default_options = {
    "member-order": "bysource",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_static_path = ["_static"] # Folder for static media lile gifs


# this generates the "[source]" links in the API reference, copied from scikit-image
linkcode_resolve = make_linkcode_resolve(
    "SCN",
    (
        "https://github.com/guillemarsan/"
        "SCN/blob/{revision}/src/"
        "{package}/{path}#L{lineno}"
    ),
)
