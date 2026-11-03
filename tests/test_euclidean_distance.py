import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from synloc import ClusterCovSampler, LocalCov, clusterCov
from synloc.kNNResampler import kNNResampler


def correlated_data():
    rng = np.random.default_rng(210)
    x = rng.normal(size=80)
    return pd.DataFrame({"x": x, "y": 7 * x + rng.normal(scale=.6, size=len(x)),
                         "z": rng.normal(size=len(x))})


@pytest.mark.parametrize("normalize", [True, False])
def test_neighbours_use_euclidean_distance_with_requested_scaling(normalize):
    data = correlated_data()
    x = data.to_numpy() / data.std().to_numpy() if normalize else data.to_numpy()
    neighbours = np.argsort(cdist(x, x), axis=1)[:, :3]
    expected = np.stack([data.iloc[index].mean().to_numpy() for index in neighbours])
    sampler = kNNResampler(data, method=lambda rows: rows.mean().to_numpy(), K=3,
                           normalize=normalize, clipping=False, n_jobs=1)
    np.testing.assert_allclose(sampler.fit().to_numpy(), expected, atol=1e-12)


@pytest.mark.parametrize("normalize", [True, False])
def test_cluster_parameters_follow_euclidean_groups_in_original_units(normalize, tmp_path):
    data = correlated_data()
    x = data.to_numpy() / data.std().to_numpy() if normalize else data.to_numpy()
    labels = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(x)
    expected = np.stack([data.iloc[labels == label].mean().to_numpy() for label in np.unique(labels)])
    model = clusterCov(data, n_clusters=4, normalize=normalize).fit_model()
    np.testing.assert_allclose(model.means_[np.argsort(model.means_[:, 0])], expected[np.argsort(expected[:, 0])])
    np.testing.assert_allclose(model.weights_ @ model.means_, data.mean())
    path = tmp_path / "mixture.npz"
    model.export_model(path)
    pd.testing.assert_frame_equal(model.sample(50, 7), ClusterCovSampler.load(path).sample(50, 7), check_exact=True)


@pytest.mark.parametrize("normalize", [True, False])
def test_singular_and_constant_columns_keep_sampling_in_original_units(normalize, tmp_path):
    data = pd.DataFrame({"money": [1000., 2000.], "copy": [2., 4.], "constant": [9., 9.]})
    model = clusterCov(data, n_clusters=1, size_min=2, normalize=normalize).fit_model()
    np.testing.assert_allclose(model.means_[0], data.mean().to_numpy())
    np.testing.assert_allclose(model.covariances_[0, :2, :2], data.iloc[:, :2].cov(), rtol=2e-5)
    assert model.sample(10, 42)["constant"].eq(9).all()
    path = tmp_path / "singular.npz"
    model.export_model(path)
    pd.testing.assert_frame_equal(model.sample(20, 2), ClusterCovSampler.load(path).sample(20, 2))
    assert np.isfinite(LocalCov(data, K=2, n_jobs=1, normalize=normalize).fit().to_numpy()).all()


def test_identical_single_record_and_final_minimum_cluster_size():
    one = pd.DataFrame({"x": [7.], "y": [2.]})
    model = clusterCov(one, n_clusters=1, size_min=1).fit_model()
    np.testing.assert_allclose(model.sample(5, 4), np.tile([7., 2.], (5, 1)))
    fitted = clusterCov(correlated_data(), n_clusters=10, size_min=15).fit_model()
    assert fitted.disclosure_diagnostics()["count"].ge(15).all()


def test_custom_neighbour_metric_remains_available():
    data = correlated_data()
    sampler = kNNResampler(data, method=lambda rows: rows.mean().to_numpy(), K=3,
                           n_jobs=1, clipping=False, Args_NearestNeighbors={"metric": "manhattan"})
    x = data.to_numpy() / data.std().to_numpy()
    neighbours = np.argsort(cdist(x, x, metric="cityblock"), axis=1)[:, :3]
    expected = np.stack([data.iloc[index].mean().to_numpy() for index in neighbours])
    np.testing.assert_allclose(sampler.fit().to_numpy(), expected, atol=1e-12)
