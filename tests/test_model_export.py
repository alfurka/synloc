import json

import numpy as np
import pandas as pd
import pytest

from synloc import ClusterCovSampler, clusterCov


def source_data():
    x = np.arange(1.0, 61.0)
    return pd.DataFrame({"earnings": 1000 * x, "hours": x, "constant": np.full(60, 0.1)})


def test_roundtrip_seeded_equality_and_parameter_only_schema(tmp_path):
    data = source_data()
    model = clusterCov(data, n_clusters=3, size_min=8)
    assert model.fit_model() is model
    assert model.synthetic is None
    path = tmp_path / "mixture.npz"
    model.export_model(path, reviewed=True)
    loaded = ClusterCovSampler.load(path)
    direct = model.sample(123, random_state=74)
    pd.testing.assert_frame_equal(direct, loaded.sample(123, random_state=74), check_exact=True)
    assert loaded.columns == tuple(data.columns)
    assert direct["constant"].eq(0.1).all()
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"version", "column_names", "means", "covariances", "weights"}
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        np.testing.assert_array_equal(archive["means"], model.means_)
        np.testing.assert_array_equal(archive["covariances"], model.covariances_)
    del model.data
    pd.testing.assert_frame_equal(model.sample(123, 74), direct, check_exact=True)


def test_moments_are_raw_units_weights_are_final_memberships():
    data = source_data()
    model = clusterCov(data, n_clusters=4, size_min=20).fit_model()
    report = model.clearance_report()
    counts = np.array([item["count"] for item in report["components"]])
    assert counts.sum() == len(data)
    assert counts.min() >= 20
    np.testing.assert_array_equal(model.weights_, counts / len(data))
    np.testing.assert_allclose(model.weights_ @ model.means_, data.mean())
    # Earnings and hours have identical standardized coordinates, not raw moments.
    np.testing.assert_allclose(model.means_[:, 0], 1000 * model.means_[:, 1])
    np.testing.assert_array_equal(model.covariances_[:, 2, :], 0)
    assert all(item["empirical_rank"] == 1 for item in report["components"])
    assert np.isfinite(model.sample(20, 0).to_numpy()).all()




def test_parameter_sampling_ignores_source_clipping():
    model = clusterCov(pd.DataFrame({"x": [0.0, 1.0]}), n_clusters=1, size_min=2, clipping=True).fit_model()
    draws = model.sample(1000, 5)
    assert ((draws.x < 0) | (draws.x > 1)).any()
    assert model.sample(0, 5).shape == (0, 1)


@pytest.mark.parametrize("n", [-1, 2.5, True, "3"])
def test_rejects_invalid_sample_sizes(n):
    sampler = ClusterCovSampler([[0]], [[[0]]], [1], ["x"])
    with pytest.raises(ValueError, match="nonnegative integer"):
        sampler.sample(n)


def test_singleton_and_singular_components_roundtrip(tmp_path):
    model = clusterCov(pd.DataFrame({"x": [7.0], "y": [0.1]}), n_clusters=1, size_min=1).fit_model()
    path = tmp_path / "singleton"
    model.export_model(path, reviewed=True)
    draws = ClusterCovSampler.load(path).sample(10, 9)
    np.testing.assert_array_equal(draws, np.tile([7, 0.1], (10, 1)))
    assert model.clearance_report()["components"][0]["empirical_rank"] == 0
    singular = ClusterCovSampler([[0, 0]], [[[1, 1], [1, 1]]], [1], ["a", "b"])
    sample = singular.sample(20, 7)
    np.testing.assert_allclose(sample.a, sample.b)


def test_clearance_is_honest_json_and_not_exported(tmp_path):
    data = pd.DataFrame({"zero": [0.0] * 4, "signed": [-1.0, 0, 1, 2], "dominated": [100.0, 0, 0, 0]})
    model = clusterCov(data, n_clusters=1, size_min=3).fit_model()
    report = model.clearance_report()
    json.dumps(report, allow_nan=False)
    assert report["mechanical_validity"]["status"] == "passed"
    assert report["privacy_approved"] is False
    assert report["required_contributor_check"]["status"] == "passed"
    dominance = report["components"][0]["dominance"]
    assert dominance["zero"] == {"status": "unassessed", "reason": "zero_total"}
    assert dominance["signed"]["status"] == "not_applicable"
    assert dominance["dominated"]["status"] == "failed"
    report["components"].clear()
    assert len(model.clearance_report()["components"]) == 1
    path = tmp_path / "research.npz"
    with pytest.raises(ValueError, match="reviewed=True"):
        model.export_model(path)
    assert not path.exists()
    with pytest.raises(ValueError, match="reviewed=True"):
        model.export_model(path, reviewed="yes")
    # Unresolved dominance review is acknowledged, not silently called approval.
    model.export_model(path, reviewed=True)
    assert model.clearance_report()["privacy_approved"] is False


def test_unconfigured_contributor_minimum_is_unassessed():
    model = clusterCov(source_data(), n_clusters=1).fit_model()
    assert model.clearance_report()["required_contributor_check"]["status"] == "unassessed"


def test_failed_required_contributor_check_blocks_even_reviewed(tmp_path, monkeypatch):
    data = source_data().iloc[:4]
    model = clusterCov(data, n_clusters=2, size_min=3)
    # Simulate final memberships violating the requirement: the export gate must
    # inspect actual memberships independently of the reassignment heuristic.
    monkeypatch.setattr(model, "_cluster_data", lambda: (
        data, data, None, np.array([0, 0, 1, 1]), pd.Series([2, 2], index=[0, 1])
    ))
    model.fit_model()
    assert model.clearance_report()["required_contributor_check"]["status"] == "failed"
    with pytest.raises(ValueError, match="contributor"):
        model.export_model(tmp_path / "blocked.npz", reviewed=True)
    assert not (tmp_path / "blocked.npz").exists()


def test_unfitted_methods_reject_use(tmp_path):
    model = clusterCov(source_data())
    for operation in [lambda: model.sample(2), model.clearance_report, lambda: model.export_model(tmp_path / "x")]:
        with pytest.raises(RuntimeError, match="fit_model"):
            operation()


def valid_payload():
    return {
        "version": np.array(1),
        "column_names": np.array(["x", "y"]),
        "weights": np.array([1.0]),
        "means": np.array([[1.0, 2.0]]),
        "covariances": np.array([[[1.0, 0.0], [0.0, 1.0]]]),
    }


@pytest.mark.parametrize("key,value", [
    ("version", np.array(2)),
    ("version", np.array([1])),
    ("version", np.array(1.0)),
    ("means", np.array([1.0, 2.0])),
    ("means", np.empty((0, 2))),
    ("means", np.array([[np.nan, 1.0]])),
    ("means", np.array([[1j, 2j]])),
    ("weights", np.array([0.9])),
    ("weights", np.array([0.0])),
    ("weights", np.array([-1.0])),
    ("weights", np.array([np.inf])),
    ("weights", np.array([[1.0]])),
    ("covariances", np.eye(2)),
    ("covariances", np.array([[[1, 0.5], [0, 1]]])),
    ("covariances", np.array([[[1, 2], [2, 1]]])),
    ("covariances", np.array([[[0, 1e-16], [1e-16, 1]]])),
    ("covariances", np.array([[[np.inf, 0], [0, 1]]])),
    ("column_names", np.array(["x", "x"])),
    ("column_names", np.array(["x", " "])),
    ("column_names", np.array(["x"])),
    ("column_names", np.array([1, 2])),
    ("column_names", np.array(["x", "y"], dtype=object)),
])
def test_rejects_invalid_export_arrays(tmp_path, key, value):
    payload = valid_payload()
    payload[key] = value
    path = tmp_path / "bad.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError):
        ClusterCovSampler.load(path)


@pytest.mark.parametrize("change", ["extra", "missing"])
def test_rejects_source_data_and_missing_fields(tmp_path, change):
    payload = valid_payload()
    if change == "extra":
        payload["source_data"] = source_data().to_numpy()
    else:
        del payload["means"]
    path = tmp_path / "bad.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="schema"):
        ClusterCovSampler.load(path)


def test_export_rejects_nonstring_column_names():
    with pytest.raises(ValueError, match="string column names"):
        clusterCov(pd.DataFrame({1: [1.0, 2.0]}), n_clusters=1).fit_model()


def test_rejects_npy_instead_of_model_archive(tmp_path):
    path = tmp_path / "not-an-archive.npy"
    np.save(path, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="NPZ"):
        ClusterCovSampler.load(path)


def test_tiny_indefinite_covariance_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        ClusterCovSampler([[0, 0]], [[[1e-20, 2e-20], [2e-20, 1e-20]]], [1], ["a", "b"])


def test_model_imputes_before_estimating_raw_moments():
    data = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "constant": [2.0] * 4})
    model = clusterCov(data, n_clusters=1, size_min=3, normalize=False).fit_model()
    np.testing.assert_array_equal(model.means_[0], [2.75, 2.0])
    dominance = model.clearance_report()["components"][0]["dominance"]
    assert dominance["constant"]["status"] == "passed"
    assert dominance["constant"]["top_one_share"] == 0.25
    assert dominance["constant"]["top_two_share"] == 0.5
