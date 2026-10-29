"""Evaluate only clusterCov on a prepared four-quarter SIPP panel.

Outputs are research diagnostics, not a disclosure approval. No clipping or
outcome transformation is used: exported moments remain in source units.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits

from synloc import ClusterCovSampler, clusterCov

COLUMNS = [f"{name}_q{quarter}" for name in ("earnings", "hours") for quarter in range(1, 5)]


def regression(y, h, fixed_effects=False):
    """OLS with quarter effects; CR0 sandwich clustered by individual.

    CR0 has no finite-sample degrees-of-freedom correction. Synthetic standard
    errors are conditional diagnostics, not synthesis-adjusted inference.
    """
    n, periods = y.shape
    quarters = np.tile(np.eye(periods)[:, 1:], (n, 1))
    if fixed_effects:
        outcome = (y - y.mean(axis=1, keepdims=True)).ravel()
        hours = (h - h.mean(axis=1, keepdims=True)).ravel()
        quarters = quarters.reshape(n, periods, -1)
        quarters = (quarters - quarters.mean(axis=1, keepdims=True)).reshape(n * periods, -1)
        design = np.column_stack((hours, quarters))
        slope_index = 0
    else:
        outcome = y.ravel()
        design = np.column_stack((np.ones(n * periods), h.ravel(), quarters))
        slope_index = 1
    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Regression is not identified for this panel.")
    residual = outcome - design @ coefficients
    scores = (design * residual[:, None]).reshape(n, periods, -1).sum(axis=1)
    bread = np.linalg.inv(design.T @ design)
    covariance = bread @ (scores.T @ scores) @ bread
    return float(coefficients[slope_index]), float(np.sqrt(max(0.0, covariance[slope_index, slope_index])))


def panel_metrics(frame):
    y = frame.iloc[:, :4].to_numpy()
    h = frame.iloc[:, 4:].to_numpy()
    result = {}
    for name, values in (("earnings", y), ("hours", h)):
        result.update({
            f"{name}_mean": float(values.mean()),
            f"{name}_sd": float(values.std(ddof=1)),
            f"{name}_zero_rate": float((values == 0).mean()),
            f"{name}_negative_rate": float((values < 0).mean()),
            f"{name}_within_variance": float(values.var(axis=1, ddof=1).mean()),
            f"{name}_between_variance": float(values.mean(axis=1).var(ddof=1)),
            f"{name}_lag1_correlation": float(np.corrcoef(values[:, :-1].ravel(), values[:, 1:].ravel())[0, 1]),
            f"{name}_change_mean": float(np.diff(values, axis=1).mean()),
            f"{name}_change_sd": float(np.diff(values, axis=1).std(ddof=1)),
        })
        for quantile in (0.1, 0.5, 0.9, 0.99):
            result[f"{name}_q{quantile:g}"] = float(np.quantile(values, quantile))
        # Only earnings is a quarterly total; hours retains its documented units.
        if name == "earnings":
            result["annual_earnings_mean"] = float(values.sum(axis=1).mean())
            result["annual_earnings_sd"] = float(values.sum(axis=1).std(ddof=1))
    result["earnings_hours_correlation"] = float(np.corrcoef(y.ravel(), h.ravel())[0, 1])
    for name, fixed in (("pooled", False), ("fixed_effects", True)):
        result[f"{name}_coefficient"], result[f"{name}_se_cr0"] = regression(y, h, fixed)
    return result


def comparison_metrics(original, synthetic):
    original_values = original.to_numpy()
    synthetic_values = synthetic.to_numpy()
    scale = original_values.std(axis=0, ddof=1)
    if (scale == 0).any():
        raise ValueError("The empirical panel must have varying outcomes in each quarter.")
    correlation_error = np.abs(np.corrcoef(original_values.T) - np.corrcoef(synthetic_values.T))
    result = {
        "mean_absolute_standardized_mean_error": float(np.mean(np.abs(synthetic_values.mean(axis=0) - original_values.mean(axis=0)) / scale)),
        "mean_absolute_sd_ratio_error": float(np.mean(np.abs(synthetic_values.std(axis=0, ddof=1) / scale - 1))),
        "mean_absolute_correlation_error": float(correlation_error[np.triu_indices(8, k=1)].mean()),
        "mean_standardized_wasserstein": float(np.mean([
            wasserstein_distance(original_values[:, j], synthetic_values[:, j]) / scale[j] for j in range(8)
        ])),
    }
    return result


def independent_reload(model_path, expected, seed):
    """Fresh interpreter receives the artifact, not the fitted object or data."""
    code = """import hashlib, sys
from synloc import ClusterCovSampler
sample = ClusterCovSampler.load(sys.argv[1]).sample(int(sys.argv[2]), random_state=int(sys.argv[3]))
print(hashlib.sha256(sample.to_numpy().tobytes()).hexdigest())
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(model_path.resolve()), str(len(expected)), str(seed)],
        check=True, capture_output=True, text=True,
    )
    expected_hash = hashlib.sha256(expected.to_numpy().tobytes()).hexdigest()
    if completed.stdout.strip() != expected_hash:
        raise AssertionError("Independent export/import sampling differs from direct sampling.")


def plot_results(replicates, destination):
    colors = ("#0072B2", "#D55E00", "#333333")
    markers = ("o", "s", "^")
    metrics = (("mean_absolute_correlation_error", "Mean absolute correlation error"),
               ("fixed_effects_error_in_original_se", "FE coefficient error / original CR0 SE"))
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, (metric, ylabel) in zip(axes, metrics):
            for index, (clusters, group) in enumerate(replicates.groupby("requested_clusters")):
                aggregate = group.groupby("minimum_size")[metric].agg(["mean", "std"])
                ax.errorbar(aggregate.index, aggregate["mean"], yerr=aggregate["std"].fillna(0),
                            marker=markers[index % len(markers)], color=colors[index % len(colors)],
                            capsize=3, label=f"{clusters} requested clusters")
            ax.axhline(0, color="0.65", linewidth=0.7)
            ax.set(xlabel="Minimum people per cluster (0 = unrestricted)", ylabel=ylabel)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("SIPP clusterCov: means ± 1 SD across synthetic replicates\nResearch exports; no configuration is certified disclosure-safe", fontsize=11)
        fig.savefig(destination / "utility_tradeoff.png", dpi=180, facecolor="white")
        fig.savefig(destination / "utility_tradeoff.pdf", facecolor="white")
        plt.close(fig)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    if list(frame.columns) != COLUMNS or len(frame) < 2 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Expected finite wide panel with exactly earnings_q1..q4, hours_q1..q4.")
    baseline = panel_metrics(frame)
    (output / "original_metrics.json").write_text(json.dumps(baseline, indent=2, allow_nan=False) + "\n")
    rows, model_rows = [], []
    for clusters in args.clusters:
        for minimum in args.minimum_sizes:
            name = f"k{clusters}_min{minimum}"
            print(f"Fitting {name}: {len(frame)} people", flush=True)
            start = time.perf_counter()
            model = clusterCov(frame, n_clusters=clusters, size_min=minimum or None, clipping=False).fit_model()
            fit_seconds = time.perf_counter() - start
            report = model.clearance_report()
            (output / f"{name}_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            path = output / f"{name}.npz"
            # Public-use research artifact. This acknowledges unresolved risks;
            # it is NOT evidence of approval to release a confidential model.
            model.export_model(path, reviewed=True)
            direct = model.sample(min(500, len(frame)), random_state=args.seed)
            loaded = ClusterCovSampler.load(path)
            pd.testing.assert_frame_equal(direct, loaded.sample(len(direct), random_state=args.seed), check_exact=True)
            independent_reload(path, direct, args.seed)
            with np.load(path, allow_pickle=False) as archive:
                realised = len(archive["weights"])
                assert set(archive.files) == {"version", "column_names", "weights", "means", "covariances"}
            model_rows.append({"configuration": name, "requested_clusters": clusters,
                               "minimum_size": minimum, "realised_clusters": realised,
                               "fit_seconds": fit_seconds, "export_bytes": path.stat().st_size,
                               "privacy_approved": False, "independent_roundtrip": True})
            for replicate in range(args.replicates):
                synthetic = loaded.sample(len(frame), random_state=args.seed + replicate)
                metrics = panel_metrics(synthetic)
                metrics.update(comparison_metrics(frame, synthetic))
                for specification in ("pooled", "fixed_effects"):
                    difference = metrics[f"{specification}_coefficient"] - baseline[f"{specification}_coefficient"]
                    metrics[f"{specification}_coefficient_difference"] = difference
                    metrics[f"{specification}_error_in_original_se"] = difference / baseline[f"{specification}_se_cr0"]
                    metrics[f"{specification}_se_ratio"] = metrics[f"{specification}_se_cr0"] / baseline[f"{specification}_se_cr0"]
                rows.append({"configuration": name, "requested_clusters": clusters,
                             "minimum_size": minimum, "replicate": replicate, **metrics})
                if replicate == 0 and clusters == args.clusters[0] and minimum == args.minimum_sizes[0]:
                    long = pd.DataFrame({"synthetic_person_id": np.repeat(np.arange(1, len(frame) + 1), 4),
                                         "quarter": np.tile(np.arange(1, 5), len(frame)),
                                         "earnings": synthetic.iloc[:, :4].to_numpy().ravel(),
                                         "hours": synthetic.iloc[:, 4:].to_numpy().ravel()})
                    assert not long.duplicated(["synthetic_person_id", "quarter"]).any()
                    long.to_csv(output / "synthetic_panel_example.csv.gz", index=False)
            pd.DataFrame(rows).to_csv(output / "replicate_metrics.csv", index=False)
            pd.DataFrame(model_rows).to_csv(output / "models.csv", index=False)
    results = pd.DataFrame(rows)
    metric_names = [name for name in results.columns if name not in ("configuration", "requested_clusters", "minimum_size", "replicate")]
    summary = results.groupby(["configuration", "requested_clusters", "minimum_size"])[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.to_csv(output / "utility_summary.csv")
    plot_results(results, output)
    manifest = {
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "people": len(frame), "quarters": 4, "columns": COLUMNS,
        "clusters": args.clusters, "minimum_sizes": args.minimum_sizes,
        "replicates": args.replicates, "seed": args.seed, "threads": args.threads,
        "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("synloc", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
        "clipping": False, "transformations": "None; variance scaling for clustering only; moments exported in original units",
        "uncertainty": "SD across independent synthetic draws conditional on each fitted model; regression SE is person-clustered CR0, not synthesis-adjusted",
        "privacy_approved": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(model_rows)} configurations and {len(rows)} synthetic replicates: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared wide.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--minimum-sizes", type=int, nargs="+", default=[0, 10, 50, 100])
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.replicates < 2 or min(args.clusters) < 1 or min(args.minimum_sizes) < 0 or args.threads < 1:
        parser.error("Use at least two replicates, positive clusters/threads, and nonnegative minimum sizes.")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)
