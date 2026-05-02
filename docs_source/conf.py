import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "synloc"
author = "Ali Furkan Kalay"
copyright = "2026, Ali Furkan Kalay"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = []
html_title = "synloc 1.0.0"

autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
