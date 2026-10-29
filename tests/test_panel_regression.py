"""Independent dummy-variable checks for the panel experiment's within estimator."""
import numpy as np
import pytest

from replication.model_export.run_experiment import regression


@pytest.mark.parametrize("fixed_effects", [False, True])
def test_panel_regression_matches_explicit_dummies_and_cluster_sandwich(fixed_effects):
    rng = np.random.default_rng(281)
    people, periods = 25, 4
    individual_effect = rng.normal(size=(people, 1))
    hours = 2 * individual_effect + rng.normal(size=(people, periods))
    earnings = 3 * hours + individual_effect + np.arange(periods) + rng.normal(size=(people, periods))
    quarter_dummies = np.tile(np.eye(periods)[:, 1:], (people, 1))
    if fixed_effects:
        intercepts = np.repeat(np.eye(people), periods, axis=0)
    else:
        intercepts = np.ones((people * periods, 1))
    design = np.column_stack((hours.ravel(), quarter_dummies, intercepts))
    inverse = np.linalg.inv(design.T @ design)
    coefficients = inverse @ design.T @ earnings.ravel()
    residuals = earnings.ravel() - design @ coefficients
    meat = np.zeros((design.shape[1], design.shape[1]))
    for person in range(people):
        rows = slice(person * periods, (person + 1) * periods)
        score = design[rows].T @ residuals[rows]
        meat += np.outer(score, score)
    expected_se = np.sqrt((inverse @ meat @ inverse)[0, 0])
    coefficient, standard_error = regression(earnings, hours, fixed_effects)
    np.testing.assert_allclose(coefficient, coefficients[0], rtol=1e-10)
    np.testing.assert_allclose(standard_error, expected_se, rtol=1e-10)


def test_fixed_effects_rejects_unidentified_time_constant_hours():
    hours = np.repeat(np.arange(1.0, 11.0)[:, None], 4, axis=1)
    earnings = hours + np.arange(4)
    with pytest.raises(ValueError, match="not identified"):
        regression(earnings, hours, fixed_effects=True)
