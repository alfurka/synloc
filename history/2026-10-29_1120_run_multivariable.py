"""Fit, export and evaluate a mixed-type clusterCov SIPP trajectory model."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from synloc.mixed_cluster import MixedClusterCovModel, MixedClusterCovSampler
from replication.model_export.multivariable.evaluate import category_levels, regress, tabulate


def hash_frame(frame):
    return hashlib.sha256(frame.to_numpy(dtype=float).tobytes()).hexdigest()


def fresh_process_check(path, direct, seed):
    code = '''import hashlib,sys
from synloc.mixed_cluster import MixedClusterCovSampler
x=MixedClusterCovSampler.load(sys.argv[1]).sample(int(sys.argv[2]),random_state=int(sys.argv[3]))
print(hashlib.sha256(x.to_numpy(dtype=float).tobytes()).hexdigest())'''
    p = subprocess.run([sys.executable, "-c", code, str(path.resolve()), str(len(direct)), str(seed)],
                       capture_output=True, text=True, check=True)
    if p.stdout.strip() != hash_frame(direct):
        raise AssertionError("Fresh-process sample differs from direct sample")


def guarded_regression(frame, levels, fixed):
    try:
        result = regress(frame, levels, fixed)
        result["status"] = "estimated"
        return result
    except ValueError as error:
        # A missing category or rank deficiency is an empirical failure, not a
        # reason to silently change the model or delete the replicate.
        return pd.DataFrame([{"term": "[unidentified]", "coefficient": np.nan,
                              "person_clustered_cr0_se": np.nan, "status": str(error)}])


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    data = pd.read_csv(input_path)
    schema_path = Path(args.schema)
    schema = json.loads(schema_path.read_text())
    expected = [item["name"] for item in schema["columns"]]
    if list(data.columns) != expected:
        raise ValueError("Prepared panel columns must match the public model schema exactly")
    if len(data) < 2 or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Prepared panel must have at least two finite records")
    print(f"Analysing {len(data)} people, {len(data.columns)} substantive/quarter columns", flush=True)
    levels = category_levels(data)
    original_stats = tabulate(data)
    original_stats.insert(0, "replicate", -1)
    original_stats.insert(0, "dataset", "original")
    original_models = []
    for fixed, name in ((False, "pooled"), (True, "fixed_effects")):
        frame = guarded_regression(data, levels, fixed)
        frame.insert(0, "model", name)
        frame.insert(0, "replicate", -1)
        frame.insert(0, "dataset", "original")
        original_models.append(frame)
    original_models = pd.concat(original_models, ignore_index=True)
    original_models.to_csv(output / "original_regressions.csv", index=False)
    if not original_models["status"].eq("estimated").all():
        raise ValueError("Original multivariable specification is not identified; inspect original_regressions.csv")
    model = MixedClusterCovModel(data, schema, n_clusters=args.clusters, size_min=args.minimum_size or None)
    start = time.perf_counter()
    model.fit_model()
    elapsed = time.perf_counter() - start
    report = model.clearance_report()
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    path = output / "mixture.npz"
    # Public-use research run only, not approval of a confidential release.
    model.export_model(path, reviewed=True)
    sampler = MixedClusterCovSampler.load(path)
    direct = model.sample(min(500, len(data)), random_state=args.seed)
    pd.testing.assert_frame_equal(direct, sampler.sample(len(direct), random_state=args.seed), check_exact=True)
    fresh_process_check(path, direct, args.seed)
    print(f"Fitted and independently reloaded {args.clusters} requested clusters in {elapsed:.1f}s", flush=True)
    stats = [original_stats]
    estimates = [original_models]
    interventions = []
    for replicate in range(args.replicates):
        seed = args.seed + replicate
        latent = sampler.raw_scores(len(data), random_state=seed)
        synthetic, diagnostic = sampler.decode(latent, return_diagnostics=True)
        check = sampler.sample(len(data), random_state=seed)
        pd.testing.assert_frame_equal(synthetic, check, check_exact=True)
        if list(synthetic.columns) != expected or not np.isfinite(synthetic.to_numpy(dtype=float)).all():
            raise ValueError("Decoded output did not preserve the finite wide panel schema")
        frame = tabulate(synthetic)
        frame.insert(0, "replicate", replicate)
        frame.insert(0, "dataset", "synthetic")
        stats.append(frame)
        for fixed, name in ((False, "pooled"), (True, "fixed_effects")):
            frame = guarded_regression(synthetic, levels, fixed)
            frame.insert(0, "model", name)
            frame.insert(0, "replicate", replicate)
            frame.insert(0, "dataset", "synthetic")
            estimates.append(frame)
        interventions.append({"replicate": replicate, "diagnostics": diagnostic})
        if replicate == 0:
            synthetic.to_csv(output / "synthetic_wide_example.csv.gz", index=False)
        print(f"Completed synthetic replicate {replicate + 1}/{args.replicates}", flush=True)
    pd.concat(stats, ignore_index=True).to_csv(output / "utility_replicates.csv", index=False)
    pd.concat(estimates, ignore_index=True).to_csv(output / "regression_replicates.csv", index=False)
    (output / "decoding_diagnostics.json").write_text(json.dumps(interventions, indent=2, allow_nan=False) + "\n")
    manifest = {"input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
                "people": len(data), "quarters": 4, "substantive_variables": 14,
                "wide_columns": len(data.columns), "requested_clusters": args.clusters,
                "minimum_contributors": args.minimum_size, "replicates": args.replicates,
                "seed": args.seed, "fit_seconds": elapsed, "model_bytes": path.stat().st_size,
                "independent_roundtrip": True, "privacy_approved": False,
                "python": sys.version, "packages": {key: importlib.metadata.version(key) for key in
                    ("synloc", "numpy", "pandas", "scipy", "scikit-learn")},
                "uncertainty": "person-clustered CR0 SE; synthetic replicates conditional on fitted model; no synthesis-adjusted inference"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed; no model is certified privacy-safe.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--minimum-size", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.clusters < 1 or args.minimum_size < 0 or args.replicates < 2 or args.threads < 1:
        parser.error("Positive clusters/threads, at least two replicates, and nonnegative minimum required")
    with threadpool_limits(limits=args.threads):
        run(args)
