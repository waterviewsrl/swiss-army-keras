# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version

# Points at the repository root (the parent of the `swiss_army_keras`
# package directory), not the package directory itself - needed so
# `automodule:: swiss_army_keras.<module>`-style directives (§11's new
# `api.rst`) can `import swiss_army_keras.<module>` by its real, dotted
# path when building from a checkout that hasn't been `pip install`ed.
# The previous `../../swiss_army_keras` pointed one level too deep (the
# package dir itself), which only supports a bare `import <module>` -
# never actually exercised, since no `.rst` file had any autodoc
# directives at all until this pass (plan.md §11).
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'swiss_army_keras'
copyright = '2022, Matteo Ferrabone'
author = 'Matteo Ferrabone'

# The full version, including alpha/beta/rc tags - read from the installed
# package's own metadata (which setuptools derives from `setup.py`'s
# `version=`) rather than a second, hand-maintained copy here, which had
# drifted 7 releases behind `setup.py` before this fix (plan.md §10/§11).
# Falls back to a literal only if the package isn't actually installed in
# the environment building the docs (e.g. building from a checkout without
# `pip install -e .` first) - kept in sync manually in that one case only.
#
# The import above is deliberately aliased (`_installed_version`, not
# `version`) - Sphinx itself reads an optional `version` config value
# straight out of this module's globals, and `from importlib.metadata
# import version` would silently shadow it with the imported *function*
# instead of a string, which Sphinx then rejects
# (`WARNING: The config value 'version' has type 'function'; expected
# 'str'.`) - caught by actually running `sphinx-build`, not just reading
# the diff.
try:
    release = _installed_version('swiss_army_keras')
except PackageNotFoundError:
    release = '0.11.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'nbsphinx'
]
source_suffix = [".rst", ".md"]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
