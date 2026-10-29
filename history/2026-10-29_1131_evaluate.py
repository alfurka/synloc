"""Evaluation of original or synthetic multivariable four-quarter panels.

All standard errors are person-clustered CR0 diagnostics, not synthesis-
adjusted inference. Every reported statistic is based on the analytical sample.
"""
import numpy as np
import pandas as pd

STABLE_CATEGORICAL = ("sex", "education", "race", "origin")
VARYING_CATEGORICAL = ("employment", "insurance", "marital")
VARYING_NUMERIC = ("earnings", "hours", "jobs", "household_size", "children", "poverty_ratio")


def panel_array(wide, name):
    return wide[[f"{name}_q{quarter}" for quarter in range(1, 5)]].to_numpy(dtype=float)

def education_group(values):
    """Predefined analysis categories; full education codes remain in the model."""
    return np.select([values <= 38, values == 39, values <= 42], [1, 2, 3], default=4)



def category_levels(frame):
    """Use original analytical-sample levels only for statistical contrasts.

    These analysis reference levels are not used to fit or decode the export.
    """
    result = {}
    for name in STABLE_CATEGORICAL:
        values = frame[name].to_numpy()
        result[name] = sorted(np.unique(education_group(values) if name == "education" else values).tolist())
    for name in VARYING_CATEGORICAL:
        result[name] = sorted(np.unique(panel_array(frame, name)).tolist())
    return result


def design(wide, levels, fixed):
    n = len(wide)
    y = panel_array(wide, "earnings")
    arrays, names = [], []

    def add(name, value):
        arrays.append(np.asarray(value, dtype=float).reshape(n, 4))
        names.append(name)

    for name in ("hours", "jobs", "household_size", "children", "poverty_ratio"):
        add(name, panel_array(wide, name))
    for name in VARYING_CATEGORICAL:
        values = panel_array(wide, name)
        for level in levels[name][1:]:
            add(f"{name}={level}", values == level)
    quarters = np.tile(np.arange(1, 5), (n, 1))
    for quarter in (2, 3, 4):
        add(f"quarter={quarter}", quarters == quarter)
    hours = panel_array(wide, "hours")
    sex = wide["sex"].to_numpy()
    for level in levels["sex"][1:]:
        add(f"sex={level}:hours", hours * (sex == level)[:, None])
    if not fixed:
        add("intercept", np.ones((n, 4)))
        add("age", np.repeat(wide["age"].to_numpy()[:, None], 4, axis=1))
        for name in STABLE_CATEGORICAL:
            values = wide[name].to_numpy()
            if name == "education":
                values = education_group(values)
            for level in levels[name][1:]:
                label = "education_group" if name == "education" else name
                add(f"{label}={level}", np.repeat((values == level)[:, None], 4, axis=1))
    x = np.stack(arrays, axis=2)
    if fixed:
        y = y - y.mean(axis=1, keepdims=True)
        x = x - x.mean(axis=1, keepdims=True)
    return y.reshape(-1), x.reshape(n * 4, -1), names


def regress(wide, levels, fixed=False, cluster_ids=None):
    """OLS with person FE (if requested), quarter FE and clustered CR0 SE.

    Standardise design columns only for numerical stability; rescale coefficients
    and covariance to original units. Fail explicitly on any unidentified term.
    """
    n = len(wide)
    y, x, names = design(wide, levels, fixed)
    scale = np.sqrt(np.mean(x * x, axis=0))
    if (scale == 0).any():
        raise ValueError(f"Unidentified zero-variation regressors: {[names[j] for j in np.flatnonzero(scale == 0)]}")
    xs = x / scale
    coef_scaled, _, rank, _ = np.linalg.lstsq(xs, y, rcond=None)
    if rank != len(names):
        raise ValueError(f"Regression is rank deficient: {rank} of {len(names)} regressors; no terms silently dropped")
    residual = y - xs @ coef_scaled
    scores = (xs * residual[:, None]).reshape(n, 4, -1).sum(axis=1)
    if cluster_ids is not None:
        ids = np.asarray(cluster_ids)
        if ids.shape != (n,) or pd.isna(ids).any():
            raise ValueError("One nonmissing cluster ID per person is required")
        _, inverse = np.unique(ids, return_inverse=True)
        grouped = np.zeros((inverse.max() + 1, len(names)))
        np.add.at(grouped, inverse, scores)
        scores = grouped
    bread = np.linalg.inv(xs.T @ xs)
    covariance = bread @ (scores.T @ scores) @ bread
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0)) / scale
    return pd.DataFrame({"term": names, "coefficient": coef_scaled / scale,
                         "person_clustered_cr0_se": standard_error})


def tabulate(wide, levels=None):
    """Long-form diagnostics, preserving all four quarters and exact categories."""
    rows = []
    if levels is None:
        levels = {name: sorted(wide[name].unique()) for name in STABLE_CATEGORICAL}
        levels.update({name: sorted(np.unique(panel_array(wide, name))) for name in VARYING_CATEGORICAL})

    def put(family, variable, detail, value, count=None):
        detail = f"{float(detail):g}" if isinstance(detail, (float, int, np.number)) else str(detail)
        rows.append({"family": family, "variable": variable, "detail": detail,
                     "value": float(value), "n": len(wide) if count is None else int(count)})

    for name in VARYING_NUMERIC:
        x = panel_array(wide, name)
        flat = x.ravel()
        for metric, value in (("mean", flat.mean()), ("sd", flat.std(ddof=1)),
                              ("zero_rate", (flat == 0).mean()), ("negative_rate", (flat < 0).mean()),
                              ("within_variance", x.var(axis=1, ddof=1).mean()),
                              ("between_variance", x.mean(axis=1).var(ddof=1)),
                              ("change_mean", np.diff(x, axis=1).mean()),
                              ("change_sd", np.diff(x, axis=1).std(ddof=1))):
            put("numeric", name, metric, value)
        for q in (0.1, 0.5, 0.9, 0.99):
            put("numeric", name, f"quantile_{q:g}", np.quantile(flat, q))
        left, right = x[:, :-1].ravel(), x[:, 1:].ravel()
        put("numeric", name, "lag1_correlation", np.corrcoef(left, right)[0, 1] if left.std() and right.std() else np.nan)
        for quarter in range(4):
            put("quarter_mean", name, quarter + 1, x[:, quarter].mean())
    put("numeric", "age", "mean", wide["age"].mean())
    put("numeric", "age", "sd", wide["age"].std(ddof=1))
    y, h = panel_array(wide, "earnings"), panel_array(wide, "hours")
    put("validity", "hours", "over_168_rate", (h > 168).mean())
    put("association", "earnings_hours", "correlation", np.corrcoef(y.ravel(), h.ravel())[0, 1])
    put("validity", "children", "above_household_rate", (panel_array(wide, "children") > panel_array(wide, "household_size")).mean())
    for name in STABLE_CATEGORICAL:
        values = wide[name].to_numpy()
        for level in levels[name]:
            put("category", name, level, (values == level).mean())
    for name in VARYING_CATEGORICAL:
        values = panel_array(wide, name)
        for level in levels[name]:
            put("category", name, level, (values == level).mean())
        for quarter in range(4):
            for level in levels[name]:
                put("category_quarter", name, f"q{quarter+1}:{level:g}", (values[:, quarter] == level).mean())
        for a in levels[name]:
            for b in levels[name]:
                first, second = values[:, :-1].ravel(), values[:, 1:].ravel()
                denom = int((first == a).sum())
                put("transition", name, f"{a:g}->{b:g}", ((first == a) & (second == b)).sum() / denom if denom else np.nan, denom)
        put("trajectory", name, "unchanged_all_quarters", (values == values[:, :1]).all(axis=1).mean())
    for name in ("jobs", "household_size", "children"):
        values = panel_array(wide, name)
        first, second = values[:, :-1], values[:, 1:]
        for label, mask in (("decrease", second < first), ("unchanged", second == first), ("increase", second > first)):
            put("count_change", name, label, mask.mean())
    zero = y == 0
    for before in (False, True):
        for after in (False, True):
            first, second = zero[:, :-1].ravel(), zero[:, 1:].ravel()
            denom = int((first == before).sum())
            put("zero_earnings_transition", "earnings", f"{int(before)}->{int(after)}",
                ((first == before) & (second == after)).sum() / denom if denom else np.nan, denom)
    employment = panel_array(wide, "employment")
    for a in levels["employment"]:
        for b in levels["employment"]:
            mask = (employment[:, :-1] == a) & (employment[:, 1:] == b)
            count = int(mask.sum())
            for outcome, values in (("earnings", y), ("hours", h)):
                put("employment_transition_change", outcome, f"{a:g}->{b:g}",
                    np.diff(values, axis=1)[mask].mean() if count else np.nan, count)
    for stratifier in (*STABLE_CATEGORICAL, "age_group"):
        if stratifier == "age_group":
            groups = np.select([wide.age < 30, wide.age < 40, wide.age < 50, wide.age < 60],
                               ["18-29", "30-39", "40-49", "50-59"], default="60-64")
        else:
            groups = wide[stratifier].to_numpy()
        for level in np.unique(groups):
            mask = groups == level
            for outcome, values in (("earnings", y), ("hours", h)):
                label = f"{level:g}" if isinstance(level, (float, int, np.number)) else str(level)
                put("subgroup", stratifier, f"{label}:{outcome}:mean", values[mask].mean(), mask.sum())
                put("subgroup", stratifier, f"{label}:{outcome}:zero_rate", (values[mask] == 0).mean(), mask.sum())
            if stratifier in ("sex", "age_group"):
                values = employment[mask]
                first, second = values[:, :-1].ravel(), values[:, 1:].ravel()
                for a in levels["employment"]:
                    denominator = int((first == a).sum())
                    for b in levels["employment"]:
                        put("subgroup_employment_transition", stratifier, f"{label}:{a:g}->{b:g}",
                            ((first == a) & (second == b)).sum() / denominator if denominator else np.nan,
                            denominator)
    return pd.DataFrame(rows)
