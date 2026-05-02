# synloc development and release notes

These notes are for local development, manual inspection, documentation builds,
and PyPI releases.

## 1. Create a clean local environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

The `.venv` folder is local to this project. Activate it whenever you want to
work on `synloc`.

## 2. Install the package for local development

Use editable mode:

```powershell
python -m pip install -e ".[dev]"
```

This installs `synloc` from the local source folder. If you edit files under
`synloc/`, Python will use those changes after you restart the Python session.
You usually do not need to reinstall after every code edit.

The `[dev]` part installs development dependencies from `pyproject.toml`.
Currently this includes:

- `pytest` for tests
- `sphinx` for documentation

It does not create release files such as wheels or source distributions.

## 2.1. If you find a mistake and edit the code

Usually you do not need to rerun:

```powershell
python -m pip install -e ".[dev]"
```

Because the package is installed in editable mode, changes inside `synloc/` are
used directly from your local source files.

After editing normal package code:

1. Save the file.
2. Restart your Python session, notebook kernel, or terminal Python process.
3. Import `synloc` again and test.

Example:

```powershell
python
```

```python
import synloc as s
```

Rerun the editable install only if you change packaging or dependency metadata,
for example:

- `pyproject.toml`
- package dependencies
- optional dependencies such as `[dev]`
- package name or version metadata
- package/module structure that packaging needs to discover

For ordinary edits to files such as `synloc/tools.py`, `synloc/dists.py`,
tests, or docs, restart Python and test again.

## 3. Play with the package interactively

After editable install:

```powershell
python
```

Example:

```python
import synloc as s

data = s.sample_trivariate_xyz(300)
resampler = s.LocalCov(data, K=20, n_jobs=1)
synthetic = resampler.fit()

resampler.comparePlots(["x"])
resampler.comparePlots(["x", "y"])
resampler.comparePlots(["x", "y", "z"])

stats = resampler.compareStats()
report = resampler.qualityReport()
```

For clustering:

```python
import synloc as s

data = s.sample_circulars_xy(300)
resampler = s.clusterCov(data, n_clusters=10, size_min=5)
synthetic = resampler.fit()

resampler.comparePlots(["x", "y"])
resampler.compareStats()
```

## 4. Run tests

Run the full test suite:

```powershell
python -m pytest tests -q
```

Run tests with printed output:

```powershell
python -m pytest tests -s
```

Run only plot-related tests:

```powershell
python -m pytest tests -q -k compareplots
```

The automated plot tests use a non-interactive Matplotlib backend, so they test
that plotting code runs without opening windows. To visually inspect plots, use
the interactive examples in section 3 instead of relying on pytest.

## 5. Build documentation

GitHub Pages publishes the `docs/` folder. Edit documentation source files in
`docs_source/`, then rebuild the HTML:

```powershell
python -m sphinx -b html docs_source docs
```

Keep this file:

```text
docs/.nojekyll
```

GitHub Pages needs `.nojekyll` so Sphinx folders such as `_static` are served
properly.

## 6. Build release artifacts

`twine` and `build` are not part of the current `[dev]` dependencies. Install
release tools separately when you are ready to publish:

```powershell
python -m pip install build twine
```

Build the package:

```powershell
python -m build
```

This creates files in `dist/`, usually:

- `.tar.gz` source distribution
- `.whl` wheel distribution

Check the built files before uploading:

```powershell
python -m twine check dist/*
```

Upload to TestPyPI first if you want a safer trial run:

```powershell
python -m twine upload --repository testpypi dist/*
```

Upload to real PyPI:

```powershell
python -m twine upload dist/*
```

## 7. Suggested release checklist

Before uploading a new version:

```powershell
python -m pytest tests -q
python -m sphinx -b html docs_source docs
python -m build
python -m twine check dist/*
```

Also check:

- `pyproject.toml` version is correct
- `synloc/__init__.py` `__version__` is correct
- `CHANGELOG.md` has a release entry
- `README.md` examples still match the public API
- `docs/.nojekyll` still exists

## 8. Important package assumptions

`synloc` expects numeric `pandas.DataFrame` inputs.

- Convert categorical variables to dummy variables before synthesis.
- Boolean dummy variables are accepted and converted to `0`/`1`.
- Missing numeric values are filled with column medians during fitting.
- Columns with only missing values are rejected.
- Non-numeric columns are rejected.
- Infinite values are rejected.

Quality metrics:

- `ks_statistic` is the Kolmogorov-Smirnov distance for one variable.
- Smaller `ks_statistic` means the original and synthetic one-dimensional
  distributions are closer.
- `qualityReport()` also reports Wasserstein distances and correlation
  difference summaries.
