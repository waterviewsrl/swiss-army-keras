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

# The full version: read textually from `swiss_army_keras/__init__.py`
# (`__version__`), the same single source of truth `setup.py` uses, so the
# docs, the package metadata and the code can never disagree. (A previous
# hand-maintained copy here drifted 7 releases behind `setup.py`.)
# `_installed_version` is kept only as a fallback; it is deliberately not
# imported as `version`, which would shadow Sphinx's own `version` config
# value with a function.
import re as _re

try:
    with open(os.path.abspath('../../swiss_army_keras/__init__.py'), encoding='utf-8') as _f:
        release = _re.search(r"^__version__ = '([^']+)'", _f.read(), _re.M).group(1)
except (OSError, AttributeError):
    try:
        release = _installed_version('swiss_army_keras')
    except PackageNotFoundError:
        release = 'unknown'
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.napoleon',
        'nbsphinx'
]
source_suffix = [".rst"]

# Single backticks in this code base's docstrings mean "code" (Markdown
# habit), not "interpreted text".
default_role = 'code'

# The checked-in notebooks carry their outputs; never re-execute them at
# build time (they train models).
nbsphinx_execute = 'never'

autodoc_member_order = 'bysource'

# Undocumented overrides (`call`, `build`, `get_config`, ...) would otherwise
# inherit Keras' own long docstrings.
autodoc_inherit_docstrings = False



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
html_static_path = []



# -- Docstring convention ----------------------------------------------------
#
# The docstrings of this package (inherited from keras-unet-collection and
# kept for every newer module) are plain text laid out as
#
#     Summary paragraph(s).
#
#     Input
#     ----------
#         name: description
#               continuation
#
#     Output
#     ----------
#         ...
#
# which is neither Google nor NumPy style, and not valid reStructuredText
# either: built as-is it produced ~265 docutils warnings/errors ("Unexpected
# indentation", "Block quote ends without a blank line", ...) and garbled
# argument lists. Rewriting several hundred docstrings is not worth the churn,
# so this hook converts the convention at build time instead: the leading
# prose stays normal text, and each `Title` + dashed-underline section becomes
# a bold title followed by a literal block that preserves the hand-made
# alignment of the argument descriptions.

_SAK_DASHES = _re.compile(r'^\s*-{3,}.*-{3,}\s*$|^\s*-{3,}\s*$')
_SAK_BULLET = _re.compile(r'^(\*|-|\d+[.)]|\(\d+\))\s+')


_SAK_STARS = _re.compile(r'(?<![`*\w])(\*{1,2}(?:\w*kwargs|args))(?![`*\w])')


def _sak_prose(lines):
    """Free prose: drop indentation, keep list items and `::` blocks well-formed."""
    out, item_indent, i = [], 0, 0
    while i < len(lines):
        raw = lines[i]
        text = raw.strip()
        if not text:
            out.append('')
            item_indent = 0
            i += 1
            continue
        # `*args` / `**kwargs` would otherwise open an emphasis span
        text = _SAK_STARS.sub(r'``\1``', text)
        bullet = _SAK_BULLET.match(text)
        if bullet:
            if out and out[-1] != '':
                out.append('')
            out.append(text)
            item_indent = len(bullet.group(0))
        else:
            out.append(' ' * item_indent + text)
        i += 1
        if text.endswith('::'):
            # an explicit literal block: keep the indented lines that follow verbatim
            base = len(raw) - len(raw.lstrip())
            block = []
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > base):
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            if block:
                ind = min(len(l) - len(l.lstrip()) for l in block if l.strip())
                out += [''] + [' ' * item_indent + '    ' + l[ind:] if l.strip() else '' for l in block] + ['']
    return out


def _sak_literal(title, body):
    """One `Title / -----` section: bold title + literal block."""
    while body and not body[0].strip():
        body = body[1:]
    while body and not body[-1].strip():
        body = body[:-1]
    out = []
    if title:
        out += ['**%s**' % title.strip(), '']
    if body:
        indent = min(len(l) - len(l.lstrip()) for l in body if l.strip())
        out += ['::', ''] + ['    ' + l[indent:] if l.strip() else '' for l in body] + ['']
    return out


def _sak_process_docstring(app, what, name, obj, options, lines):
    if not name.startswith('swiss_army_keras') or not lines:
        return
    first = next((i for i, l in enumerate(lines) if _SAK_DASHES.match(l)), None)
    if first is None:
        lines[:] = _sak_prose(lines)
        return
    # a dashed line directly under a non-blank line is that line's underline
    has_title = first > 0 and lines[first - 1].strip() != ''
    head = lines[:first - 1] if has_title else lines[:first]
    out = _sak_prose(head) + ['']
    title = lines[first - 1] if has_title else None
    body = []
    i = first + 1
    while i < len(lines):
        if _SAK_DASHES.match(lines[i]):
            if body and body[-1].strip():      # underline of a new section title
                new_title = body.pop()
                out += _sak_literal(title, body)
                title, body = new_title, []
            else:                              # free-standing separator line
                out += _sak_literal(title, body)
                title, body = None, []
        else:
            body.append(lines[i])
        i += 1
    out += _sak_literal(title, body)
    lines[:] = out


def setup(app):
    app.connect('autodoc-process-docstring', _sak_process_docstring, priority=100)
