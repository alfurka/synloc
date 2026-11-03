import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synloc as s


def test_clustercov_fit_returns_requested_size_and_metrics():
    np.random.seed(123)
    data = s.sample_circulars_xy(60)
    resampler = s.clusterCov(data=data, n_clusters=6, size_min=4)

    synthetic = resampler.fit(sample_size=20)

    assert resampler.fitted is True
    assert resampler.synthetic is synthetic
    assert synthetic.shape == (20, 2)
    assert list(synthetic.columns) == list(data.columns)
    assert not synthetic.isna().any().any()
    assert np.isfinite(synthetic.to_numpy()).all()
    assert len(resampler.data_distances) == len(data)
    assert len(resampler.synthetic_distances) == len(synthetic)
    assert np.isfinite(resampler.data_distances).all()
    assert np.isfinite(resampler.synthetic_distances).all()
    assert "ks_statistic" in resampler.compareStats().columns
    assert "overall" in resampler.qualityReport()


def test_clustercov_compareplots_runs_after_fit(monkeypatch):
    np.random.seed(123)
    data = s.sample_circulars_xy(30)
    resampler = s.clusterCov(data=data, n_clusters=4, size_min=3)
    resampler.fit(sample_size=12)

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    resampler.comparePlots("x")
    resampler.comparePlots(["x", "y"])

    assert len(shown) == 2
    plt.close("all")


def test_clustercov_rejects_bad_cluster_arguments():
    data = s.sample_circulars_xy(10)

    with pytest.raises(ValueError, match="n_clusters"):
        s.clusterCov(data=data, n_clusters=0)

    with pytest.raises(ValueError, match="size_min"):
        s.clusterCov(data=data, n_clusters=2, size_min=0)


def test_clustercov_rejects_impossible_size_min_on_fit():
    data = s.sample_circulars_xy(10)
    resampler = s.clusterCov(data=data, n_clusters=2, size_min=11)

    with pytest.raises(ValueError, match="size_min"):
        resampler.fit()


def test_clustercov_preserves_cluster_constant_columns_without_clipping():
    np.random.seed(123)
    data = pd.DataFrame({
        "position": [0.0, 0.1, 100.0, 100.1, 200.0, 200.1],
        "cluster_constant": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "paired_value": [7.0, 7.0, 8.0, 8.0, 9.0, 9.0],
    })
    resampler = s.clusterCov(data=data, n_clusters=3, size_min=2, clipping=False)

    synthetic = resampler.fit()

    assert synthetic["cluster_constant"].isin([1.0, 2.0, 3.0]).all()
    assert set(synthetic["cluster_constant"]) == {1.0, 2.0, 3.0}


@pytest.mark.parametrize("integer_type", [np.int32, np.int64, np.uint64])
def test_cluster_accepts_numpy_integer_counts(integer_type):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    sampler = s.clusterCov(data, n_clusters=integer_type(2), size_min=integer_type(2))

    assert sampler.fit(sample_size=integer_type(3)).shape == (3, 1)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False), 1.5])
@pytest.mark.parametrize("parameter", ["n_clusters", "size_min", "sample_size"])
def test_cluster_rejects_noninteger_counts(parameter, value):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    kwargs = {"n_clusters": 2, "size_min": 2}
    if parameter != "sample_size":
        kwargs[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        sampler = s.clusterCov(data, **kwargs)
        if parameter == "sample_size":
            sampler.fit(value)


def _repeat_cluster_mean(rows, size):
    return pd.DataFrame(np.repeat([rows.mean().to_numpy()], size, axis=0), columns=rows.columns)


@pytest.mark.parametrize("failure", ["callback", "diagnostics"])
def test_cluster_failed_refit_clears_fitted_results(failure):
    data = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    sampler = s.clusterResampler(data, _repeat_cluster_mean, n_clusters=2)
    sampler.fit(3)

    def invalid_method(rows, size):
        if failure == "callback":
            raise ValueError("callback failure")
        return pd.DataFrame(np.nan, index=range(size), columns=rows.columns)

    sampler.method = invalid_method
    with pytest.raises(ValueError):
        sampler.fit(3)
    assert sampler.fitted is False
    assert sampler.synthetic is None
    assert sampler.metrics is None
    assert sampler.data_distances is None
    assert sampler.synthetic_distances is None
    assert sampler.compareStats() is None
    assert sampler.qualityReport() is None


def test_cluster_missing_values_remain_imputed_for_repeat_fit_and_diagnostics():
    data = pd.DataFrame({"x": [1.0, np.nan, 4.0, 8.0], "y": [2.0, 3.0, 1.0, 5.0]})
    original = data.copy()
    sampler = s.clusterResampler(data, _repeat_cluster_mean, n_clusters=2)
    first = sampler.fit()
    pd.testing.assert_frame_equal(sampler.fit(), first)
    pd.testing.assert_frame_equal(data, original)
    pd.testing.assert_frame_equal(sampler.data, data.fillna(data.median()))
    report = sampler.qualityReport()
    pd.testing.assert_frame_equal(sampler.metrics["per_variable"], report["per_variable"])
    pd.testing.assert_series_equal(sampler.metrics["overall"], report["overall"])
