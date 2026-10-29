"""Prepare the public 2018 SIPP's 2017 quarterly earnings/hours panel.

See DATA_PREPARATION.md for source definitions and selection limitations.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "SSUID", "PNUM", "SPANEL", "SWAVE", "MONTHCODE", "RIN_UNIV",
    "TAGE", "RMNUMJOBS", "TPEARN", "TMWKHRS",
]
KEYS = ["SSUID", "PNUM"]
WIDE_COLUMNS = [f"{variable}_q{quarter}" for variable in ("earnings", "hours")
                for quarter in range(1, 5)]
REFERENCE_YEAR = 2017
SOURCE_URL = (
    "https://www2.census.gov/programs-surveys/sipp/data/datasets/2018/pu2018.csv.gz"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_panel(input_path, output_dir, *, chunksize=25000):
    """Write wide.csv, monthly long.csv and metadata.json; return metadata.

    Read only ten columns in chunks, then retain the narrow person-month table
    in memory. No network access or survey weights are needed at runtime.
    Duplicate person-months or an unexpected panel/wave are fatal errors, not
    silently deduplicated or pooled. Output directories must be dedicated to
    this preparation; existing output files are replaced only after validation.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    source_checksum = _sha256(input_path)
    dtypes = {column: "float64" for column in COLUMNS if column not in KEYS}
    dtypes.update({column: "string" for column in KEYS})
    chunks = []
    with pd.read_csv(
        input_path, sep="|", usecols=COLUMNS, dtype=dtypes,
        na_values=["", "."], keep_default_na=False, chunksize=chunksize,
        compression="infer",
    ) as reader:
        for chunk in reader:
            chunks.append(chunk)
    if not chunks:
        raise ValueError("Input contains no person-month records")
    frame = pd.concat(chunks, ignore_index=True)
    del chunks
    if frame.empty:
        raise ValueError("Input contains no person-month records")
    if frame[KEYS].isna().any().any():
        raise ValueError("Person identifiers must not be missing")
    if not frame["SSUID"].str.fullmatch(r"[0-9]{14}").all():
        raise ValueError("SSUID must be the official 14-digit sample identifier")
    if not frame["PNUM"].str.fullmatch(r"[0-9]{3}").all():
        raise ValueError("PNUM must be the official three-digit person number")
    if not (frame["SPANEL"].eq(2018) & frame["SWAVE"].eq(1)).all():
        raise ValueError("Expected only the 2018 SIPP panel, wave 1 (reference 2017)")
    if not frame["MONTHCODE"].isin(range(1, 13)).all():
        raise ValueError("MONTHCODE must contain integer reference months 1 through 12")
    frame["MONTHCODE"] = frame["MONTHCODE"].astype("int8")
    duplicates = int(frame.duplicated(KEYS + ["MONTHCODE"], keep=False).sum())
    if duplicates:
        raise ValueError(f"Found {duplicates} records with duplicate person-month keys")

    # Numeric NIU is blank or '.' in this release. Only a documented no-job
    # month licenses turning a missing labor outcome into a structural zero.
    no_job = frame["RMNUMJOBS"].eq(0)
    for source, label in (("TPEARN", "earnings"), ("TMWKHRS", "hours")):
        flag = f"_{label}_structural_zero"
        frame[flag] = no_job & frame[source].isna()
        frame.loc[frame[flag], source] = 0.0
    frame["_in_frame"] = frame["RIN_UNIV"].eq(1)
    frame["_working_age"] = frame["TAGE"].between(18, 64)
    frame["_valid_labor"] = (
        np.isfinite(frame["TPEARN"])
        & np.isfinite(frame["TMWKHRS"])
        & frame["TMWKHRS"].between(0, 99.99)
    )
    persons = frame.groupby(KEYS, sort=True).agg(
        months=("MONTHCODE", "size"),
        in_frame=("_in_frame", "all"),
        working_age=("_working_age", "all"),
        valid_labor=("_valid_labor", "all"),
    )
    flow = [{"stage": "input", "persons": len(persons), "person_months": len(frame),
             "excluded_persons": 0, "excluded_person_months": 0}]
    retained = pd.Series(True, index=persons.index)
    criteria = [
        ("twelve_unique_calendar_months", persons["months"].eq(12)),
        ("in_survey_frame_all_months", persons["in_frame"]),
        ("age_18_to_64_all_months", persons["working_age"]),
        ("complete_valid_labor_values_all_months", persons["valid_labor"]),
    ]
    for stage, criterion in criteria:
        removed = retained & ~criterion
        retained &= criterion
        flow.append({
            "stage": stage,
            "persons": int(retained.sum()),
            "person_months": int(persons.loc[retained, "months"].sum()),
            "excluded_persons": int(removed.sum()),
            "excluded_person_months": int(persons.loc[removed, "months"].sum()),
        })
    selected = frame.loc[frame.set_index(KEYS).index.isin(persons.index[retained])].copy()
    if selected.empty:
        raise ValueError("No people satisfy the balanced working-age panel definition")
    selected.sort_values(KEYS + ["MONTHCODE"], inplace=True)
    selected["quarter"] = (selected["MONTHCODE"] - 1) // 3 + 1
    quarterly = selected.groupby(KEYS + ["quarter"], sort=True).agg(
        earnings=("TPEARN", "sum"), hours=("TMWKHRS", "mean"),
        months=("MONTHCODE", "size"),
    )
    if not quarterly["months"].eq(3).all():
        raise ValueError("Every retained person-quarter must contain exactly three months")
    if not quarterly.groupby(level=KEYS).size().eq(4).all():
        raise ValueError("Every retained person must have four quarters")
    wide = quarterly[["earnings", "hours"]].unstack("quarter")
    wide.columns = [f"{variable}_q{quarter}" for variable, quarter in wide.columns]
    wide = wide[WIDE_COLUMNS]
    if not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Quarterly output must contain only finite numeric values")
    long = selected[KEYS + ["MONTHCODE", "quarter", "TAGE", "RMNUMJOBS", "TPEARN",
                           "TMWKHRS", "_earnings_structural_zero", "_hours_structural_zero"]].rename(
        columns={"MONTHCODE": "month", "TAGE": "age", "RMNUMJOBS": "number_of_jobs",
                 "TPEARN": "earnings", "TMWKHRS": "hours",
                 "_earnings_structural_zero": "earnings_structural_zero",
                 "_hours_structural_zero": "hours_structural_zero"}
    )
    long.insert(2, "year", REFERENCE_YEAR)
    source_metadata = json.loads(Path(__file__).with_name("sipp_source_metadata.json").read_text())
    metadata = {
        "preparation_version": 1,
        "source": {"filename": input_path.name, "sha256": source_checksum,
                   "size_bytes": input_path.stat().st_size, "official_url": SOURCE_URL,
                   "panel": 2018, "wave": 1, "reference_year": REFERENCE_YEAR,
                   "release_identity": "Local input identified by SHA-256; not authenticated against a Census file hash"},
        "source_metadata": source_metadata,
        "selected_columns": COLUMNS,
        "wide_columns": WIDE_COLUMNS,
        "sample_flow": flow,
        "retained_persons": len(wide),
        "retained_person_months": len(selected),
        "retained_person_quarters": len(quarterly),
        "structural_zero_fills": {
            label: {"input_person_months": int(frame[f"_{label}_structural_zero"].sum()),
                    "retained_person_months": int(selected[f"_{label}_structural_zero"].sum())}
            for label in ("earnings", "hours")
        },
        "retained_negative_earnings_person_months": int(selected["TPEARN"].lt(0).sum()),
        "retained_negative_earnings_persons": int(
            selected.loc[selected["TPEARN"].lt(0), KEYS].drop_duplicates().shape[0]),
        "input_invalid_labor_person_months_after_zero_fill": int((~frame["_valid_labor"]).sum()),
        "definitions": {
            "population": "Age 18-64 and RIN_UNIV=1 in every one of 12 unique months",
            "earnings": "Quarterly sum of TPEARN, nominal 2017 dollars, including business profits/losses",
            "hours": "Equal-month mean of three TMWKHRS values, hours/week; not total quarterly hours",
            "structural_zeros": "Fill missing TPEARN/TMWKHRS only when RMNUMJOBS=0; never overwrite observed values",
            "missing": "Blank and '.' numeric values are missing; other parse failures stop preparation",
            "nonnegative_filter": "None for earnings: legitimate negative profits/losses retained; hours require 0..99.99",
            "transformations": "No logs, winsorization, inflation adjustment, scaling, or clipping",
            "weights": "Unweighted person panel; survey and replicate weights are not used",
            "order": "Rows sorted by SSUID then PNUM; wide drops identifiers, long retains them",
        },
        "limitations": [
            "Balanced and complete-case selection excludes partial-year/out-of-frame and missing-outcome people; not representative of all adults.",
            "Requiring ages 18-64 throughout excludes people turning 18 or 65 during the reference year.",
            "Public-use earnings are Census-processed, may be imputed/topcoded, and are not unmodified confidential values.",
            "TPEARN varies with month length; this deliberately does not use the smoothed TPEARN_ALT.",
            "Monthly weekly-hours rates are averaged equally across months, not reconstructed as actual quarter totals.",
            "No employment/positive-earnings requirement: nonworkers and zero months remain; business losses remain.",
            "Public-use source clearance does not confer privacy approval on a subsequently fitted model.",
        ],
        "software": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_dir / "wide.csv", index=False)
    long.to_csv(output_dir / "long.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Official pipe-delimited pu2018.csv[.gz]")
    parser.add_argument("--output", type=Path, required=True, help="Directory for prepared CSVs and metadata")
    parser.add_argument("--chunksize", type=int, default=25000)
    args = parser.parse_args()
    metadata = prepare_panel(args.input, args.output, chunksize=args.chunksize)
    print(json.dumps({"retained_persons": metadata["retained_persons"],
                      "sample_flow": metadata["sample_flow"]}, indent=2))


if __name__ == "__main__":
    main()
