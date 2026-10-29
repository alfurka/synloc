import numpy as np
import pandas as pd
import pytest

from replication.model_export.multivariable.evaluate import category_levels, regress, tabulate


def example():
    rng = np.random.default_rng(42)
    n = 80
    sex = np.tile([1, 2], n // 2)
    data = {"sex": sex, "age": rng.integers(18, 65, n), "education": rng.choice([1, 2], n),
            "race": rng.choice([1, 2], n), "origin": rng.choice([1, 2], n)}
    effect = rng.normal(0, 10, n)
    for q in range(1, 5):
        hours = rng.uniform(10, 50, n)
        jobs = rng.integers(0, 3, n)
        size = rng.integers(2, 6, n)
        children = rng.integers(0, 2, n)
        poverty = rng.uniform(0, 5, n)
        employment = rng.integers(1, 3, n)
        insurance = rng.integers(1, 3, n)
        marital = rng.integers(1, 3, n)
        data.update({f"hours_q{q}": hours, f"jobs_q{q}": jobs, f"household_size_q{q}": size,
                     f"children_q{q}": children, f"poverty_ratio_q{q}": poverty,
                     f"employment_q{q}": employment, f"insurance_q{q}": insurance,
                     f"marital_q{q}": marital,
                     f"earnings_q{q}": 3 * hours + 4 * jobs + (sex == 2) * hours + q + effect})
    return pd.DataFrame(data)


def test_fixed_effects_recovers_within_coefficients_and_interaction():
    frame = example()
    result = regress(frame, category_levels(frame), fixed=True).set_index("term")
    assert result.loc["hours", "coefficient"] == pytest.approx(3, abs=1e-9)
    assert result.loc["jobs", "coefficient"] == pytest.approx(4, abs=1e-9)
    assert result.loc["sex=2:hours", "coefficient"] == pytest.approx(1, abs=1e-9)
    assert "sex=2" not in result.index


def test_unidentified_within_term_does_not_silently_disappear():
    frame = example()
    for q in range(1, 5):
        frame[f"jobs_q{q}"] = 1
    with pytest.raises(ValueError, match="Unidentified"):
        regress(frame, category_levels(frame), fixed=True)


def test_transition_denominators_and_subgroup_sizes():
    frame = example()
    for q in range(1, 5):
        frame[f"insurance_q{q}"] = 1 if q < 3 else 2
    stats = tabulate(frame).set_index(["family", "variable", "detail"])
    assert stats.loc[("transition", "insurance", "1->2"), "value"] == pytest.approx(0.5)
    assert stats.loc[("transition", "insurance", "1->2"), "n"] == 2 * len(frame)
    assert stats.loc[("subgroup", "sex", "1:earnings:mean"), "n"] == len(frame) // 2
    assert stats.loc[("observed_spell_count_per_person", "insurance", "1:2"), "value"] == 1
    assert stats.loc[("observed_spell_count_per_person", "insurance", "2:2"), "value"] == 1
