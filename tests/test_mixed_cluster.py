import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from synloc.mixed_cluster import MixedClusterCovModel, MixedClusterCovSampler


def schema():
    return {
        "version": 1,
        "columns": [
            {"name": "sex", "kind": "category", "categories": [2, 1, 3], "labels": ["two", "one", "three"]},
            {"name": "age", "kind": "integer", "min": 18, "max": 64},
            {"name": "earnings", "kind": "continuous", "min": None, "max": None},
            {"name": "hours", "kind": "continuous", "min": 0, "max": 100},
            {"name": "household", "kind": "integer", "min": 1, "max": 20},
            {"name": "children", "kind": "integer", "min": 0, "max": 20},
        ],
        "constraints": [{"kind": "less_equal", "left": "children", "right": "household"}],
    }


def data():
    return pd.DataFrame({
        "sex": [1, 2, 1, 2, 1, 2], "age": [18, 24, 30, 40, 50, 64],
        "earnings": [-100, 0, 100, 200, 300, 400], "hours": [0, 10, 20, 30, 40, 50],
        "household": [1, 2, 3, 4, 4, 5], "children": [0, 0, 1, 2, 3, 4],
    })


def fitted():
    return MixedClusterCovModel(data(), schema(), n_clusters=1, size_min=2).fit_model()


def test_roundtrip_encoded_constants_and_public_only_archive(tmp_path):
    model = fitted()
    path = tmp_path / "mixed.npz"
    with pytest.raises(ValueError, match="reviewed=True"):
        model.export_model(path)
    model.export_model(path, reviewed=True)
    sampler = MixedClusterCovSampler.load(path)
    pd.testing.assert_frame_equal(model.raw_scores(100, 17), sampler.raw_scores(100, 17), check_exact=True)
    direct, diag = model.sample(100, 17, return_diagnostics=True)
    loaded, loaded_diag = sampler.sample(100, 17, return_diagnostics=True)
    pd.testing.assert_frame_equal(direct, loaded, check_exact=True)
    assert diag == loaded_diag
    assert set(direct.sex) <= {1, 2, 3}
    assert (direct.children <= direct.household).all()
    assert direct.age.between(18, 64).all()
    assert direct.hours.between(0, 100).all()
    assert model.raw_scores(100, 17)["score_2"].eq(0).all()
    assert sampler.sample(0, 0).shape == (0, 6)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"version", "metadata", "means", "covariances", "weights"}
        assert json.loads(str(archive["metadata"].item())) == schema()
        assert archive["means"].shape == (1, 8)
    assert model.clearance_report()["privacy_approved"] is False


def test_deterministic_ties_rounding_clipping_and_interventions():
    # One constant component makes raw values straightforward to inspect.
    sampler = MixedClusterCovSampler(np.zeros((1, 8)), np.zeros((1, 8, 8)), [1], schema())
    raw = pd.DataFrame([
        [1, 1, 0, 18.5, -250.0, -1.0, 2.5, 4.5],
        [0, 1, 0, 99.0, -300.0, 120.0, 1.0, -1.0],
    ], columns=[f"score_{i}" for i in range(8)])
    result, diag = sampler.decode(raw, return_diagnostics=True)
    assert result.sex.tolist() == [2, 1]
    assert result.age.tolist() == [18, 64]
    assert result.earnings.tolist() == [-250, -300]
    assert result.hours.tolist() == [0, 100]
    assert result.household.tolist() == [2, 1]
    assert result.children.tolist() == [2, 0]
    assert diag["columns"]["sex"]["invalid_one_hot_before"] == 1
    assert diag["columns"]["sex"]["ties"] == 1
    assert diag["columns"]["sex"]["invalid_category_after"] == 0
    assert diag["columns"]["age"]["noninteger_before"] == 1
    assert diag["columns"]["age"]["outside_bounds_before"] == 1
    assert diag["columns"]["age"]["outside_bounds_after"] == 0
    assert diag["constraints"]["children<=household"] == {
        "raw_violations_before": 1, "violations_before": 1, "changed": 1, "violations_after": 0,
    }


def test_independent_process_loads_without_source(tmp_path):
    model = fitted()
    path = tmp_path / "mixed.npz"
    model.export_model(path, reviewed=True)
    expected = model.sample(10, 123).to_json(orient="split", double_precision=15)
    script = ("import sys; from synloc.mixed_cluster import MixedClusterCovSampler; "
              "print(MixedClusterCovSampler.load(sys.argv[1]).sample(10,123).to_json(orient='split',double_precision=15))")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", script, str(path)], cwd=tmp_path, env=env,
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == json.loads(expected)


@pytest.mark.parametrize("change", [
    lambda s: s.update(extra=True),
    lambda s: s.update(version=2),
    lambda s: s["columns"][0].update(categories=[1, 1]),
    lambda s: s["columns"][0].update(categories=[]),
    lambda s: s["columns"][0].update(labels=["only one"]),
    lambda s: s["columns"][0].update(labels=["", "one", "three"]),
    lambda s: s["columns"][0].update(categories=[1, None]),
    lambda s: s["columns"][0].update(transform="learned"),
    lambda s: s["columns"][1].update(min=None),
    lambda s: s["columns"][1].update(max=17),
    lambda s: s["columns"][1].update(kind="log"),
    lambda s: s["columns"][1].update(name="sex"),
    lambda s: s["constraints"][0].update(kind="unknown"),
    lambda s: s["constraints"][0].update(right="missing"),
])
def test_schema_rejects_unsupported_or_invalid_metadata(change):
    bad = schema()
    change(bad)
    with pytest.raises(ValueError):
        MixedClusterCovSampler(np.zeros((1, 8)), np.zeros((1, 8, 8)), [1], bad)


@pytest.mark.parametrize("mutation", [
    lambda d: d.assign(sex=9),
    lambda d: d.assign(age=np.nan),
    lambda d: d.assign(age=12),
    lambda d: d.assign(age=18.5),
    lambda d: d.assign(children=20),
    lambda d: d.drop(columns="sex"),
])
def test_rejects_invalid_source(mutation):
    with pytest.raises(ValueError):
        MixedClusterCovModel(mutation(data()), schema())


def test_invalid_weights_covariance_and_scores():
    with pytest.raises(ValueError, match="weights"):
        MixedClusterCovSampler(np.zeros((1, 8)), np.zeros((1, 8, 8)), [0], schema())
    cov = np.eye(8)[None, :, :]
    cov[0, 0, 0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        MixedClusterCovSampler(np.zeros((1, 8)), cov, [1], schema())
    sampler = MixedClusterCovSampler(np.zeros((1, 8)), np.zeros((1, 8, 8)), [1], schema())
    raw = sampler.raw_scores(2, 1)
    raw.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sampler.decode(raw)
    with pytest.raises(ValueError):
        sampler.raw_scores(-1, 1)


def test_constant_component_and_single_public_category():
    public = {"version": 1, "columns": [
        {"name": "status", "kind": "category", "categories": [7], "labels": ["seven"]},
        {"name": "earnings", "kind": "continuous", "min": None, "max": None},
    ], "constraints": []}
    sampler = MixedClusterCovSampler([[1, -42]], np.zeros((1, 2, 2)), [1], public)
    raw = sampler.raw_scores(5, 7)
    assert raw["score_0"].eq(1).all()
    assert raw["score_1"].eq(-42).all()
    decoded = sampler.decode(raw)
    assert decoded["status"].eq(7).all()
    assert decoded["earnings"].eq(-42).all()


def test_bad_archives_fail_closed(tmp_path):
    path = tmp_path / "bad.npz"
    payload = {"version": np.array(1), "metadata": np.array(json.dumps(schema())),
               "means": np.zeros((1, 8)), "covariances": np.zeros((1, 8, 8)), "weights": np.ones(1)}
    for mutation in (lambda p: p.update(secret=np.array([1])),
                     lambda p: p.update(version=np.array(2)),
                     lambda p: p.update(metadata=np.array('{"version":1,"version":1}')),
                     lambda p: p.update(metadata=np.array(json.dumps({**schema(), "transform": "log"}))),
                     lambda p: p.update(metadata=np.array([object()], dtype=object)),
                     lambda p: p.update(weights=np.array([0.])),
                     lambda p: p.update(means=np.zeros((1, 7)))):
        changed = copy.deepcopy(payload)
        mutation(changed)
        np.savez_compressed(path, **changed)
        with pytest.raises(ValueError):
            MixedClusterCovSampler.load(path)


def test_gate_blocks_failed_contributor_minimum(tmp_path, monkeypatch):
    model = fitted()
    monkeypatch.setitem(model._fit._clearance["required_contributor_check"], "status", "failed")
    with pytest.raises(ValueError, match="contributor"):
        model.export_model(tmp_path / "blocked.npz", reviewed=True)
    assert not (tmp_path / "blocked.npz").exists()
