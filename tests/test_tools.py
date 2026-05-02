import numpy as np
import pandas as pd
import pytest

from synloc.tools import (
    compareStats,
    compute_k_distances,
    kolmogorov_distances,
    new_cluster_sizes,
    quality_report,
    validate_numeric_dataframe,
)


def test_compute_k_distances_handles_small_samples_and_large_k():
    one_row = pd.DataFrame({"x": [1.0], "y": [2.0]})
    many_rows = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]})

    assert np.array_equal(compute_k_distances(one_row, K=5), np.zeros(1))
    distances = compute_k_distances(many_rows, K=99)
    assert distances.shape == (3,)
    assert np.isfinite(distances).all()


def test_new_cluster_sizes_sum_to_requested_size_and_are_nonnegative():
    original = pd.Series([10, 5, 1], index=[0, 1, 2])

    resized = new_cluster_sizes(original, 4)

    assert resized.sum() == 4
    assert (resized >= 0).all()
    assert list(resized.index) == list(original.index)


def test_compare_stats_and_quality_report_include_distribution_metrics():
    original = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 1.5, 3.0]})
    synthetic = pd.DataFrame({"x": [0.1, 1.1, 1.9], "y": [0.9, 1.4, 3.1]})

    stats = compareStats(original, synthetic)
    ks = kolmogorov_distances(original, synthetic)
    report = quality_report(original, synthetic)

    assert "ks_statistic" in stats.columns
    assert "wasserstein_distance" in stats.columns
    assert list(ks.index) == ["x", "y"]
    assert report["overall"]["mean_ks_statistic"] >= 0


def test_compare_stats_handles_identical_constant_columns():
    original = pd.DataFrame({"constant": [4.0, 4.0, 4.0]})
    synthetic = pd.DataFrame({"constant": [4.0, 4.0, 4.0]})

    stats = compareStats(original, synthetic)

    assert stats.loc["constant", "original_std"] == 0
    assert stats.loc["constant", "synthetic_std"] == 0
    assert stats.loc["constant", "std_difference"] == 0
    assert stats.loc["constant", "wasserstein_distance"] == 0


def test_validate_numeric_dataframe_accepts_bool_and_rejects_bad_values():
    data = pd.DataFrame({"x": [1.0, 2.0], "dummy": [True, False]})
    cleaned = validate_numeric_dataframe(data)

    assert cleaned["dummy"].tolist() == [1, 0]

    with pytest.raises(TypeError, match="non-numeric"):
        validate_numeric_dataframe(pd.DataFrame({"x": ["a", "b"]}))

    with pytest.raises(ValueError, match="only missing"):
        validate_numeric_dataframe(pd.DataFrame({"x": [np.nan, np.nan]}))
