"""Summarize replicated results and predeclared diagnostic gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(output):
    output = Path(output)
    metrics = pd.read_csv(output / "utility_replicates.csv", dtype={"detail": str})
    base = metrics[metrics.dataset.eq("original")][["family", "variable", "detail", "value", "n"]].rename(columns={"value": "original", "n": "original_n"})
    draws = metrics[metrics.dataset.eq("synthetic")]
    aggregate = draws.groupby(["family", "variable", "detail"], dropna=False).value.agg(["mean", "std", "count"]).reset_index()
    table = base.merge(aggregate, on=["family", "variable", "detail"], how="outer")
    table["difference"] = table["mean"] - table["original"]
    table.to_csv(output / "utility_comparison.csv", index=False)

    regs = pd.read_csv(output / "regression_replicates.csv")
    orig = regs[regs.dataset.eq("original")][["model", "term", "coefficient", "person_clustered_cr0_se", "status"]].rename(
        columns={"coefficient": "original_coefficient", "person_clustered_cr0_se": "original_se", "status": "original_status"})
    syn = regs[regs.dataset.eq("synthetic") & regs.status.eq("estimated")].groupby(["model", "term"]).agg(
        synthetic_coefficient_mean=("coefficient", "mean"), synthetic_coefficient_sd=("coefficient", "std"),
        synthetic_se_mean=("person_clustered_cr0_se", "mean"), estimated_replicates=("replicate", "nunique")).reset_index()
    comparison = orig.merge(syn, on=["model", "term"], how="outer")
    comparison["difference"] = comparison.synthetic_coefficient_mean - comparison.original_coefficient
    comparison["difference_in_original_se"] = comparison.difference / comparison.original_se
    comparison["se_ratio"] = comparison.synthetic_se_mean / comparison.original_se
    comparison.to_csv(output / "regression_comparison.csv", index=False)
    expected = int(regs[regs.dataset.eq("synthetic")].replicate.nunique())

    def table_fail(mask, tolerance, require_all=True):
        selected = table.loc[mask].copy()
        fails = selected[(selected.difference.abs() > tolerance) | selected["mean"].isna() |
                         (selected["count"] < expected if require_all else False)]
        return {"checked": len(selected), "failed": len(fails),
                "failures": fails[["family", "variable", "detail", "original", "mean", "count"]].replace({np.nan: None}).to_dict("records")}

    prevalence = table_fail(((table.family.eq("category") & table.original.ge(0.01)) |
                             (table.family.eq("numeric") & table.variable.isin(["earnings", "hours"]) & table.detail.eq("zero_rate"))), 0.05)
    transitions = table_fail(table.family.eq("transition") & table.original_n.ge(200) & table.original.notna(), 0.10)
    nonestimated = regs[~regs.status.eq("estimated")][["dataset", "replicate", "model", "status"]]
    identifiable = {"failed": len(nonestimated), "failures": nonestimated.to_dict("records")}
    coefficient_fail = comparison[(comparison.original_se.gt(0)) & ((comparison.difference_in_original_se.abs() > 2) | comparison.synthetic_coefficient_mean.isna() | (comparison.estimated_replicates < expected))]
    coefficient_gate = {"checked": int(comparison.original_se.gt(0).sum()), "failed": len(coefficient_fail),
                        "failures": coefficient_fail[["model", "term", "difference_in_original_se", "estimated_replicates"]].replace({np.nan: None}).to_dict("records")}
    key = comparison[comparison.model.eq("fixed_effects") & (comparison.term.isin(["hours", "jobs", "household_size", "children", "poverty_ratio"]) | comparison.term.str.contains(":hours", regex=False, na=False))]
    keyfail = key[(~key.se_ratio.between(0.75, 1.25)) | (key.estimated_replicates < expected)]
    se_gate = {"checked": len(key), "failed": len(keyfail),
               "failures": keyfail[["term", "original_se", "synthetic_se_mean", "se_ratio"]].replace({np.nan: None}).to_dict("records")}
    diagnostics = json.loads((output / "decoding_diagnostics.json").read_text())
    invalid_after = []
    for item in diagnostics:
        for name, checks in item["diagnostics"]["columns"].items():
            for field, count in checks.items():
                if field in ("invalid_category_after", "outside_bounds_after", "noninteger_after") and count:
                    invalid_after.append({"replicate": item["replicate"], "column": name, "field": field, "count": count})
        for name, checks in item["diagnostics"]["constraints"].items():
            if checks["violations_after"]:
                invalid_after.append({"replicate": item["replicate"], "constraint": name, "count": checks["violations_after"]})
    audit = json.loads((output / "audit.json").read_text())
    contributor_status = audit["required_contributor_check"]["status"]
    gates = {"validity": {"failed": len(invalid_after), "failures": invalid_after},
             "category_and_zero_prevalence": prevalence, "transitions": transitions,
             "identification": identifiable, "coefficient": coefficient_gate, "standard_errors": se_gate,
             "contributor_check": {"status": contributor_status, "failed": int(contributor_status != "passed")},
             "privacy_approved": False, "interpretation": "Research diagnostic gates only; no privacy clearance or inference guarantee."}
    gates["initial_candidate_viable"] = all(item.get("failed", 0) == 0 for item in gates.values() if isinstance(item, dict))
    (output / "diagnostic_gates.json").write_text(json.dumps(gates, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: (value.get("failed") if isinstance(value, dict) else value) for key, value in gates.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    summarize(parser.parse_args().output)
