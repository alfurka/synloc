"""Evaluate a pooled linear probability model for positive quarterly earnings.

This is descriptive, not causal; group-clustered CR0 standard errors on
synthetic data do not account for estimating the synthesis model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from synloc.mixed_cluster import MixedClusterCovSampler
from replication.model_export.multivariable.evaluate import category_levels, regress


def probability_model(wide, levels):
    panel = wide.copy()
    for q in range(1, 5):
        name = f"earnings_q{q}"
        panel[name] = (panel[name] > 0).astype(float)
    result = regress(panel, levels, fixed=False)
    return result


def main(args):
    original = pd.read_csv(args.input)
    levels = category_levels(original)
    reference = probability_model(original, levels)
    reference.insert(0, "replicate", -1)
    reference.insert(0, "dataset", "original")
    all_estimates = [reference]
    sampler = MixedClusterCovSampler.load(args.model)
    for replicate in range(args.replicates):
        synthetic = sampler.sample(len(original), random_state=args.seed + replicate)
        try:
            estimate = probability_model(synthetic, levels)
            estimate["status"] = "estimated"
        except ValueError as error:
            estimate = pd.DataFrame([{"term": "[unidentified]", "coefficient": np.nan,
                                      "person_clustered_cr0_se": np.nan, "status": str(error)}])
        estimate.insert(0, "replicate", replicate)
        estimate.insert(0, "dataset", "synthetic")
        all_estimates.append(estimate)
    result = pd.concat(all_estimates, ignore_index=True)
    result.loc[result.dataset.eq("original"), "status"] = "estimated"
    result.to_csv(Path(args.output) / "positive_earnings_regressions.csv", index=False)
    print(f"Positive-earnings LPM: {args.replicates} synthetic draws; "
          f"{(result.status != 'estimated').sum()} unidentified model runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(args.threads):
        main(args)
