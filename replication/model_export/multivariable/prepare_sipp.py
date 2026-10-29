"""Prepare an unweighted, balanced 2017 SIPP panel; see DATA_PREPARATION.md.

Run: python prepare_sipp.py --input PATH/pu2018.csv.gz --output PATH/prepared
The outputs are internal person-level data and must not be released as model metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["SSUID", "PNUM"]
COLUMNS = KEYS + [
    "SHHADID", "SPANEL", "SWAVE", "MONTHCODE", "RIN_UNIV", "TAGE_EHC",
    "ESEX", "EEDUC", "ERACE", "EORIGIN", "RMESR", "RMNUMJOBS",
    "RHLTHMTH", "EMS_EHC", "RHNUMPER", "RHNUMU18", "TFINCPOV", "RFPOV",
    "TPEARN", "TMWKHRS",
]
STATIC = {"sex": "ESEX", "age": "TAGE_EHC", "education": "EEDUC", "race": "ERACE", "origin": "EORIGIN"}
QUARTERLY = {"earnings": "TPEARN", "hours": "TMWKHRS", "employment": "RMESR", "jobs": "RMNUMJOBS", "insurance": "RHLTHMTH", "marital": "EMS_EHC", "household_size": "RHNUMPER", "children": "RHNUMU18", "poverty_ratio": "TFINCPOV"}
CATEGORIES = {
    "ESEX": {1: "Male", 2: "Female"},
    "EEDUC": dict(zip(range(31, 47), ["Less than first grade", "First to fourth grade", "Fifth or sixth grade", "Seventh or eighth grade", "Ninth grade", "Tenth grade", "Eleventh grade", "Twelfth grade without diploma", "High school graduate or equivalent", "Some college, less than one year", "One or more years of college, no degree", "Associate's degree", "Bachelor's degree", "Master's degree", "Professional degree", "Doctorate"])),
    "ERACE": {1: "White alone", 2: "Black alone", 3: "Asian alone", 4: "Residual"},
    "EORIGIN": {1: "Hispanic, Spanish or Latino", 2: "Not Hispanic, Spanish or Latino"},
    "RMESR": {1: "Job entire month, worked all weeks", 2: "Job all month, unpaid absence not due to layoff", 3: "Job all month, unpaid absence due to layoff", 4: "Job some weeks, no layoff or job search", 5: "Job some weeks, some layoff or job search", 6: "No job, layoff or job search all weeks", 7: "No job, layoff or job search some weeks", 8: "No job, no layoff or job search"},
    "RHLTHMTH": {1: "Insured", 2: "Not insured"},
    "EMS_EHC": {1: "Married, spouse present", 2: "Married, spouse absent", 3: "Widowed", 4: "Divorced", 5: "Separated", 6: "Never married"},
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_number(series, lower, upper, *, integer=False):
    valid = np.isfinite(series) & series.between(lower, upper)
    return valid & series.eq(np.floor(series)) if integer else valid


def prepare_panel(input_path, output_dir, *, chunksize=25000):
    """Write wide.csv, long.csv and metadata.json, returning internal metadata.

    IDs occur only in long.csv. Schema.json is public, fixed and independent of
    the input; sample counts and diagnostics are internal and require review.
    """
    input_path, output_dir = Path(input_path), Path(output_dir)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    dtypes = {column: "float64" for column in COLUMNS if column not in KEYS + ["SHHADID"]}
    dtypes.update({column: "string" for column in KEYS + ["SHHADID"]})
    chunks = list(pd.read_csv(input_path, sep="|", usecols=COLUMNS, dtype=dtypes,
                              na_values=["", "."], keep_default_na=False,
                              chunksize=chunksize, compression="infer"))
    if not chunks:
        raise ValueError("Input contains no person-month records")
    frame = pd.concat(chunks, ignore_index=True)
    del chunks
    if frame.empty or frame[KEYS].isna().any().any():
        raise ValueError("Input must contain records with person identifiers")
    if not frame.SSUID.str.fullmatch(r"[0-9]{14}").all() or not frame.PNUM.str.fullmatch(r"[0-9]{3}").all():
        raise ValueError("Expected 14-digit SSUID and three-digit PNUM")
    if not (frame.SPANEL.eq(2018) & frame.SWAVE.eq(1)).all():
        raise ValueError("Expected only 2018 panel, wave 1")
    if not frame.MONTHCODE.isin(range(1, 13)).all():
        raise ValueError("MONTHCODE must be an integer in 1..12")
    if frame.duplicated(KEYS + ["MONTHCODE"]).any():
        raise ValueError("Duplicate person-month keys")
    frame = frame.sort_values(KEYS + ["MONTHCODE"], kind="stable").reset_index(drop=True)
    group = frame.groupby(KEYS, sort=True)
    persons = group.size().to_frame("months")
    # First observed month is used for subgroup flow even for incomplete people;
    # December education is used when available, otherwise explicitly missing.
    first = group.nth(0).set_index(KEYS)
    december = frame.loc[frame.MONTHCODE.eq(12)].set_index(KEYS)
    profiles = first[["ESEX", "TAGE_EHC", "ERACE", "EORIGIN"]].rename(columns={"ESEX": "sex", "TAGE_EHC": "age", "ERACE": "race", "EORIGIN": "origin"}).reindex(persons.index)
    profiles["education"] = december["EEDUC"].reindex(persons.index)
    age = profiles["age"]
    profiles["age_group"] = pd.cut(age, [-np.inf, 17, 24, 34, 44, 54, 64, np.inf], labels=["under_18", "18_24", "25_34", "35_44", "45_54", "55_64", "65_plus"]).astype("object")

    is_end = frame.MONTHCODE.isin([3, 6, 9, 12])
    is_dec = frame.MONTHCODE.eq(12)
    # Two independent recodes must agree before an NIU labor value is a zero.
    no_job = frame.RMNUMJOBS.eq(0) & frame.RMESR.isin([6, 7, 8])
    fills = {}
    for col in ["TPEARN", "TMWKHRS"]:
        flag = f"_fill_{col}"
        frame[flag] = no_job & frame[col].isna()
        fills[col] = flag
        frame.loc[frame[flag], col] = 0.0

    valid = {
        "earnings": _valid_number(frame.TPEARN, -99999999, 99999999, integer=True),
        "hours": _valid_number(frame.TMWKHRS, 0, 99.99),
        "jobs": _valid_number(frame.RMNUMJOBS, 0, 9, integer=True),
        "employment": frame.RMESR.isin(CATEGORIES["RMESR"]),
        "insurance": frame.RHLTHMTH.isin(CATEGORIES["RHLTHMTH"]),
        "marital": frame.EMS_EHC.isin(CATEGORIES["EMS_EHC"]),
        "household_size": _valid_number(frame.RHNUMPER, 1, 20, integer=True),
        "children": _valid_number(frame.RHNUMU18, 0, 20, integer=True) & frame.RHNUMU18.le(frame.RHNUMPER),
        "poverty_ratio": _valid_number(frame.TFINCPOV, 0, 999.999) & _valid_number(frame.RFPOV, 1, 7000, integer=True),
    }
    for name, source in STATIC.items():
        if name != "age":
            valid[name] = frame[source].isin(CATEGORIES[source])
    criteria = [
        ("twelve_unique_calendar_months", persons.months.eq(12)),
        ("in_survey_frame_all_months", frame.RIN_UNIV.eq(1).groupby([frame[k] for k in KEYS]).all()),
        ("age_18_to_64_all_months", _valid_number(frame.TAGE_EHC, 18, 64, integer=True).groupby([frame[k] for k in KEYS]).all()),
    ]
    for name in ["sex", "race", "origin"]:
        source = STATIC[name]
        constant = group[source].nunique(dropna=False).eq(1)
        criteria.append((f"valid_constant_{name}", valid[name].groupby([frame[k] for k in KEYS]).all() & constant))
    criteria.append(("valid_december_education", (valid["education"] | ~is_dec).groupby([frame[k] for k in KEYS]).all()))
    for name in QUARTERLY:
        required = pd.Series(True, index=frame.index) if name in ("earnings", "hours") else is_end
        criteria.append((f"valid_{name}", (valid[name] | ~required).groupby([frame[k] for k in KEYS]).all()))

    def composition(mask):
        result = {}
        for col in ["sex", "age_group", "education", "race", "origin"]:
            values = profiles.loc[mask, col].astype(object).fillna("missing")
            result[col] = {str(k): int(v) for k, v in values.value_counts(dropna=False).items()}
        return result

    retained = pd.Series(True, index=persons.index)
    flow = [{"stage": "input", "persons": len(persons), "person_months": len(frame), "excluded_persons": 0, "excluded_person_months": 0, "retained_characteristics": composition(retained)}]
    missingness = {}
    for name, check in valid.items():
        required = is_dec if name == "education" else (pd.Series(True, index=frame.index) if name in ("sex", "race", "origin", "earnings", "hours") else is_end)
        bad = required & ~check
        col = STATIC.get(name, QUARTERLY.get(name))
        missingness[name] = {"required_person_months": int(required.sum()), "missing_person_months": int((required & frame[col].isna()).sum()), "invalid_or_missing_person_months": int(bad.sum()), "affected_persons": int(bad.groupby([frame[k] for k in KEYS]).any().sum())}
    for stage, criterion in criteria:
        criterion = criterion.reindex(persons.index, fill_value=False)
        removed = retained & ~criterion
        retained &= criterion
        flow.append({"stage": stage, "persons": int(retained.sum()), "person_months": int(persons.loc[retained, "months"].sum()), "excluded_persons": int(removed.sum()), "excluded_person_months": int(persons.loc[removed, "months"].sum()), "excluded_characteristics": composition(removed), "retained_characteristics": composition(retained)})
    selected = frame.loc[frame.set_index(KEYS).index.isin(persons.index[retained])].copy()
    if selected.empty:
        raise ValueError("No people satisfy the balanced complete-case panel definition")
    selected["quarter"] = ((selected.MONTHCODE - 1) // 3 + 1).astype(int)
    grouped = selected.groupby(KEYS + ["quarter"], sort=True)
    labor = grouped.agg(earnings=("TPEARN", "sum"), hours=("TMWKHRS", "mean"), months=("MONTHCODE", "size"))
    if not labor.months.eq(3).all() or not labor.groupby(level=KEYS).size().eq(4).all():
        raise ValueError("Expected exactly three months in each of four quarters")
    end = selected.loc[selected.MONTHCODE.isin([3, 6, 9, 12])].set_index(KEYS + ["quarter"])
    quarterly = labor[["earnings", "hours"]].join(end[[source for name, source in QUARTERLY.items() if name not in ("earnings", "hours")]].rename(columns={v: k for k, v in QUARTERLY.items()}))
    base = selected.loc[selected.MONTHCODE.eq(1)].set_index(KEYS)[["ESEX", "TAGE_EHC", "ERACE", "EORIGIN"]].rename(columns={v: k for k, v in STATIC.items()})
    base["education"] = selected.loc[selected.MONTHCODE.eq(12)].set_index(KEYS)["EEDUC"]
    base = base[list(STATIC)]
    wide_q = quarterly[list(QUARTERLY)].unstack("quarter")
    wide_q.columns = [f"{name}_q{quarter}" for name, quarter in wide_q.columns]
    wide = base.join(wide_q)
    if not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite prepared values")
    long = quarterly.reset_index().merge(base.reset_index(), on=KEYS, validate="many_to_one")
    household = end[["SHHADID"]].reset_index()
    long = long.merge(household, on=KEYS + ["quarter"], validate="one_to_one")
    long["person_id"] = long.SSUID + ":" + long.PNUM
    long["household_id"] = long.SSUID + ":" + long.SHHADID
    long = long[["person_id", "household_id", "quarter"] + list(STATIC) + list(QUARTERLY)]
    metadata = {
        "preparation_version": 1,
        "source": {"filename": input_path.name, "sha256": _sha256(input_path), "size_bytes": input_path.stat().st_size, "url": "https://www2.census.gov/programs-surveys/sipp/data/datasets/2018/pu2018.csv.gz", "panel": 2018, "wave": 1, "reference_year": 2017, "release_identity": "Local SHA-256 not authenticated against an official Census hash"},
        "selected_columns": COLUMNS, "source_by_feature": STATIC | QUARTERLY,
        "public_category_labels": CATEGORIES, "wide_columns": wide.columns.tolist(),
        "sample_flow": flow, "missingness_before_selection": missingness,
        "structural_zero_fills": {col: {"input_person_months": int(frame[flag].sum()), "retained_person_months": int(selected[flag].sum())} for col, flag in fills.items()},
        "no_job_recode_disagreement_person_months": int((frame.RMNUMJOBS.eq(0) & frame.RMESR.isin([1, 2, 3, 4, 5])).sum()),
        "retained_negative_earnings_person_months": int(selected.TPEARN.lt(0).sum()),
        "retained_persons": len(wide), "retained_person_months": len(selected), "retained_person_quarters": len(long),
        "household_id_missing_quarters": int(long.household_id.isna().sum()),
        "definitions": {"age": "January TAGE_EHC, age 18–64 in all 12 months", "education": "December EEDUC, not necessarily known earlier", "earnings": "sum of 3 TPEARN monthly dollars, allowing losses", "hours": "equal-month mean of 3 TMWKHRS rates in hours/week", "other_quarterly": "last calendar month of quarter", "zero_fill": "only missing labor amounts with RMNUMJOBS=0 and RMESR in 6,7,8; observed values never overwritten", "poverty": "TFINCPOV excluding Type 2; require positive RFPOV; no rescaling", "weights": "unweighted", "selection": "balanced all-month frame/age and complete required values; see sample flow", "household": "internal SSUID:SHHADID at quarter end; not a synthetic household identifier"},
        "software": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_dir / "wide.csv", index=False)
    long.to_csv(output_dir / "long.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=25000)
    args = parser.parse_args()
    metadata = prepare_panel(args.input, args.output, chunksize=args.chunksize)
    print(json.dumps({"retained_persons": metadata["retained_persons"], "sample_flow": metadata["sample_flow"]}, indent=2))


if __name__ == "__main__":
    main()
