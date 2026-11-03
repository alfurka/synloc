import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from synloc import ClusterCovSampler, LocalCov, clusterCov


def data():
    return pd.DataFrame({"x": [1., 2., 5., 11., 19., 34.],
                         "y": [2., 4., 7., 14., 26., 51.],
                         "constant": [3.] * 6})


@pytest.mark.parametrize("normalize", [False, True])
def test_selected_components_use_full_data_and_original_units(normalize):
    frame = data()
    proposal = LocalCov(frame, K=3, normalize=normalize, n_jobs=1).prepare_export(2, random_state=17)
    anchors = np.random.default_rng(17).choice(len(frame), size=2, replace=False)
    scale = frame.std().replace(0, 1) if normalize else 1
    transformed = (frame / scale).to_numpy()
    neighbours = np.argsort(cdist(transformed[anchors], transformed), axis=1)[:, :3]
    expected_means = np.stack([frame.iloc[indices].mean().to_numpy() for indices in neighbours])
    np.testing.assert_allclose(proposal.means_, expected_means)
    np.testing.assert_array_equal(proposal.weights_, [0.5, 0.5])
    for j, indices in enumerate(neighbours):
        raw_cov = frame.iloc[indices, :2].cov().to_numpy()
        jitter = max(np.trace(raw_cov) / 2, 1.0) * 1e-10
        np.testing.assert_allclose(proposal.covariances_[j, :2, :2], raw_cov + np.eye(2) * jitter)
        np.testing.assert_array_equal(proposal.covariances_[j, 2], 0)
    assert [item['count'] for item in proposal.clearance_report()['components']] == [3, 3]
    assert proposal.clearance_report()['required_contributor_check']['status'] == 'unassessed'


def test_export_roundtrip_and_report_are_for_fixed_selection(tmp_path):
    frame = data()
    model = LocalCov(frame, K=3, n_jobs=1)
    proposal = model.prepare_export(4, random_state=9, minimum_contributors=3)
    expected = proposal.sample(250, random_state=7)
    report = proposal.clearance_report()
    assert report['privacy_approved'] is False
    assert report['required_contributor_check']['status'] == 'passed'
    assert len(report['components']) == 4
    report['components'].clear()
    assert len(proposal.clearance_report()['components']) == 4
    path = tmp_path / 'model.npz'
    with pytest.raises(ValueError, match='reviewed=True'):
        proposal.export_model(path)
    assert not path.exists()
    del model, frame
    proposal.export_model(path, reviewed=True)
    loaded = ClusterCovSampler.load(path)
    pd.testing.assert_frame_equal(loaded.sample(250, random_state=7), expected, check_exact=True)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {'version', 'column_names', 'weights', 'means', 'covariances'}
        assert archive['means'].shape == (4, 3)


def test_all_components_default_and_singleton_neighbourhoods():
    frame = data()
    proposal = LocalCov(frame, K=1, n_jobs=1).prepare_export()
    np.testing.assert_array_equal(proposal.means_, frame.to_numpy())
    np.testing.assert_array_equal(proposal.covariances_, np.zeros((len(frame), 3, 3)))
    np.testing.assert_allclose(proposal.weights_, np.full(len(frame), 1 / len(frame)))
    assert all(item['count'] == 1 for item in proposal.clearance_report()['components'])
    assert proposal.clearance_report()['components'][0]['dominance']['x']['status'] == 'failed'


def test_actual_neighbour_count_and_contributor_gate(tmp_path):
    proposal = LocalCov(data(), K=30, n_jobs=1).prepare_export(2, random_state=0, minimum_contributors=7)
    report = proposal.clearance_report()
    assert report['required_contributor_check']['status'] == 'failed'
    assert report['required_contributor_check']['failed_components'] == [0, 1]
    assert [component['count'] for component in report['components']] == [6, 6]
    path = tmp_path / 'blocked.npz'
    with pytest.raises(ValueError, match='contributor'):
        proposal.export_model(path, reviewed=True)
    assert not path.exists()


@pytest.mark.parametrize('count', [0, -1, 7, 1.5, True, '2'])
def test_rejects_invalid_component_counts(count):
    with pytest.raises(ValueError, match='n_components'):
        LocalCov(data(), K=2, n_jobs=1).prepare_export(count)


@pytest.mark.parametrize('minimum', [0, -1, 2.5, True, '2'])
def test_rejects_invalid_minimum_contributors(minimum):
    with pytest.raises(ValueError, match='minimum_contributors'):
        LocalCov(data(), K=2, n_jobs=1).prepare_export(2, minimum_contributors=minimum)


def test_custom_metric_and_missing_value_imputation():
    frame = pd.DataFrame({'x': [0., 1., 3., 8.], 'y': [0., 5., np.nan, 1.]})
    imputed = frame.fillna(frame.median())
    proposal = LocalCov(frame, K=2, normalize=False, n_jobs=1,
                        Args_NearestNeighbors={'metric': 'manhattan'}).prepare_export()
    distances = cdist(imputed.to_numpy(), imputed.to_numpy(), metric='cityblock')
    indices = np.argsort(distances, axis=1)[:, :2]
    expected = np.stack([imputed.iloc[row].mean().to_numpy() for row in indices])
    np.testing.assert_allclose(proposal.means_, expected)


def test_export_sampling_does_not_clip_to_source_bounds():
    proposal = LocalCov(pd.DataFrame({'x': [0., 1.]}), K=2, clipping=True, n_jobs=1).prepare_export()
    sample = proposal.sample(1000, random_state=4)
    assert ((sample.x < 0) | (sample.x > 1)).any()


@pytest.mark.parametrize("varying", [False, True])
@pytest.mark.parametrize("method", ["local", "cluster"])
def test_legacy_and_export_preserve_decimal_constants(method, varying):
    frame = pd.DataFrame({"constant": [0.1] * 3})
    if varying:
        frame["varying"] = [1., 2., 4.]
    if method == "local":
        model = LocalCov(frame, K=3, n_jobs=1, clipping=False)
        fitted = model.prepare_export()
    else:
        model = clusterCov(frame, n_clusters=1, clipping=False)
        fitted = model.fit_model()
    assert model.fit()["constant"].eq(0.1).all()
    assert fitted.sample(20, random_state=4)["constant"].eq(0.1).all()


def test_proposal_parameters_cannot_diverge_from_reviewed_export(tmp_path):
    proposal = LocalCov(data(), K=3, n_jobs=1).prepare_export(2, random_state=7)
    expected = proposal.sample(20, random_state=4)
    for name in ("means_", "covariances_", "weights_"):
        parameters = getattr(proposal, name)
        with pytest.raises(ValueError):
            parameters.flat[0] = 100
        with pytest.raises(AttributeError):
            setattr(proposal, name, np.zeros_like(parameters))
    with pytest.raises(AttributeError):
        proposal.columns = ("wrong",)
    path = tmp_path / "fixed.npz"
    proposal.export_model(path, reviewed=True)
    pd.testing.assert_frame_equal(
        expected, ClusterCovSampler.load(path).sample(20, random_state=4),
        check_exact=True,
    )
