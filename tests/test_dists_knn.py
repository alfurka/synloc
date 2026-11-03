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


@pytest.mark.parametrize("integer_type", [np.int32, np.int64, np.uint64])
def test_knn_accepts_numpy_integer_counts(integer_type):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    resampler = s.LocalCov(data, K=integer_type(2), n_jobs=integer_type(1))

    assert resampler.fit(sample_size=integer_type(3)).shape == (3, 1)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False), 1.5])
@pytest.mark.parametrize("parameter", ["K", "n_jobs", "sample_size"])
def test_knn_rejects_noninteger_counts(parameter, value):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    kwargs = {"K": 2, "n_jobs": 1}
    if parameter != "sample_size":
        kwargs[parameter] = value
    error = TypeError if parameter == "n_jobs" else ValueError
    with pytest.raises(error, match=parameter):
        resampler = s.LocalCov(data, **kwargs)
        if parameter == "sample_size":
            resampler.fit(value)


def _random_knn_point(rows):
    return rows.mean().to_numpy() + np.random.normal(size=rows.shape[1])


@pytest.mark.parametrize("sampler_class", [s.kNNResampler, s.LocalCov])
@pytest.mark.parametrize("sample_size", [None, 7])
def test_knn_seed_reproduces_draws_across_workers_and_threading_override(sampler_class, sample_size):
    from joblib import parallel_backend

    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0], "y": [3.0, 6.0, 1.0, 4.0]})
    kwargs = {"method": _random_knn_point} if sampler_class is s.kNNResampler else {}
    serial = sampler_class(data, K=3, clipping=False, n_jobs=1, random_state=19, **kwargs)
    expected = serial.fit(sample_size)
    pd.testing.assert_frame_equal(serial.fit(sample_size), expected, check_exact=True)
    parallel = sampler_class(data, K=3, clipping=False, n_jobs=2, random_state=19, **kwargs)
    with parallel_backend("threading"):
        actual = parallel.fit(sample_size)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    other = sampler_class(data, K=3, clipping=False, n_jobs=1, random_state=20, **kwargs)
    assert not other.fit(sample_size).equals(expected)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_knn_seed_does_not_advance_global_numpy_rng(n_jobs):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    state = np.random.get_state()
    try:
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        sampler = s.kNNResampler(data, _random_knn_point, K=3, n_jobs=n_jobs, random_state=19)
        sampler.fit(7)
        np.testing.assert_array_equal(np.random.random(5), expected)
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("sample_size", [None, 7])
def test_knn_none_seed_preserves_serial_global_rng_behavior(sample_size):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    state = np.random.get_state()
    try:
        np.random.seed(123)
        anchors = np.arange(len(data)) if sample_size is None else np.random.choice(len(data), sample_size)
        expected = pd.DataFrame([_random_knn_point(data.iloc[[i]]) for i in anchors], columns=data.columns)
        expected_next = np.random.random(5)
        np.random.seed(123)
        sampler = s.kNNResampler(data, _random_knn_point, K=1, clipping=False, n_jobs=1)
        pd.testing.assert_frame_equal(sampler.fit(sample_size), expected, check_exact=True)
        np.testing.assert_array_equal(np.random.random(5), expected_next)
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("failure", ["callback", "diagnostics"])
def test_knn_failed_refit_clears_fitted_results_and_restores_rng(failure):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    sampler = s.kNNResampler(data, _random_knn_point, K=2, n_jobs=1, random_state=19)
    sampler.fit(3)

    def invalid_method(rows):
        np.random.random()
        if failure == "callback":
            raise ValueError("callback failure")
        return np.full(rows.shape[1], np.nan)

    sampler.method = invalid_method
    state = np.random.get_state()
    try:
        np.random.seed(123)
        expected = np.random.random(5)
        np.random.seed(123)
        with pytest.raises((RuntimeError, ValueError)):
            sampler.fit(3)
        np.testing.assert_array_equal(np.random.random(5), expected)
    finally:
        np.random.set_state(state)
    assert sampler.fitted is False
    assert sampler.synthetic is None
    assert sampler.metrics is None
    assert sampler.data_distances is None
    assert sampler.synthetic_distances is None
    assert sampler.compareStats() is None
    assert sampler.qualityReport() is None


def test_knn_missing_values_remain_imputed_for_repeat_fit_and_diagnostics():
    data = pd.DataFrame({"x": [1.0, np.nan, 4.0, 8.0], "y": [2.0, 3.0, 1.0, 5.0]})
    original = data.copy()
    sampler = s.kNNResampler(data, lambda rows: rows.mean().to_numpy(), K=2, n_jobs=1)
    first = sampler.fit()
    pd.testing.assert_frame_equal(sampler.fit(), first)
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(sampler.data, data.fillna(data.median()))
    report = sampler.qualityReport()
    pd.testing.assert_frame_equal(sampler.metrics["per_variable"], report["per_variable"])
    pd.testing.assert_series_equal(sampler.metrics["overall"], report["overall"])
