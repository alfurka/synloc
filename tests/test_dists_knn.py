import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synloc as s


def test_sample_helpers_return_numeric_dataframes():
    data3d = s.sample_trivariate_xyz(25)
    data2d = s.sample_circulars_xy(25)

    assert data3d.shape == (25, 3)
    assert data2d.shape == (25, 2)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in data3d.dtypes)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in data2d.dtypes)


def test_localcov_fit_returns_requested_size_and_metrics():
    np.random.seed(123)
    data = s.sample_trivariate_xyz(40)
    resampler = s.LocalCov(data=data, K=5, n_jobs=1)

    synthetic = resampler.fit(sample_size=25)

    assert resampler.fitted is True
    assert resampler.synthetic is synthetic
    assert synthetic.shape == (25, 3)
    assert list(synthetic.columns) == list(data.columns)
    assert not synthetic.isna().any().any()
    assert np.isfinite(synthetic.to_numpy()).all()
    assert len(resampler.data_distances) == len(data)
    assert len(resampler.synthetic_distances) == len(synthetic)
    assert np.isfinite(resampler.data_distances).all()
    assert np.isfinite(resampler.synthetic_distances).all()

    stats = resampler.compareStats()
    assert set(["ks_statistic", "wasserstein_distance"]).issubset(stats.columns)
    report = resampler.qualityReport()
    assert "per_variable" in report
    assert "overall" in report
    assert "mean_ks_statistic" in report["overall"].index


def test_localcov_compareplots_runs_after_fit(monkeypatch):
    np.random.seed(123)
    data = s.sample_trivariate_xyz(20)
    resampler = s.LocalCov(data=data, K=4, n_jobs=1)
    resampler.fit(sample_size=10)

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    resampler.comparePlots("x")
    resampler.comparePlots(["x", "y"])
    resampler.comparePlots(["x", "y", "z"])

    assert len(shown) == 3
    plt.close("all")


def test_localcov_rejects_categorical_columns():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "category": ["a", "b", "c"]})

    with pytest.raises(TypeError, match="non-numeric"):
        s.LocalCov(data=data, K=2, n_jobs=1)


def test_localcov_imputes_missing_values_and_accepts_bool_dummies():
    np.random.seed(123)
    data = pd.DataFrame({
        "x": [1.0, np.nan, 3.0, 4.0],
        "dummy": [True, False, True, False],
    })
    resampler = s.LocalCov(data=data, K=2, n_jobs=1)

    synthetic = resampler.fit(sample_size=4)

    assert not resampler.data.isna().any().any()
    assert set(resampler.data["dummy"].unique()).issubset({0, 1})
    assert synthetic.shape == (4, 2)


def test_localcov_preserves_locally_constant_columns_without_clipping():
    np.random.seed(123)
    data = pd.DataFrame({
        "position": [0.0, 0.01, 100.0, 100.01, 200.0, 200.01],
        "local_constant": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "paired_value": [7.0, 7.0, 8.0, 8.0, 9.0, 9.0],
    })
    resampler = s.LocalCov(data=data, K=2, n_jobs=1, clipping=False)

    synthetic = resampler.fit()

    assert synthetic["local_constant"].isin([1.0, 2.0, 3.0]).all()
    assert set(synthetic["local_constant"]) == {1.0, 2.0, 3.0}


def test_localcov_rejects_infinite_values():
    data = pd.DataFrame({"x": [1.0, np.inf, 3.0]})

    with pytest.raises(ValueError, match="infinity"):
        s.LocalCov(data=data, K=1, n_jobs=1)
