import json

import numpy as np
import pandas as pd
import pytest

from replication.model_export.multivariable.prepare_sipp import prepare_panel


def fixture():
    rows = []
    for person in range(101, 106):
        for month in range(1, 13):
            rows.append({"SSUID": "00000000000001", "PNUM": str(person), "SHHADID": "011",
                         "SPANEL": 2018, "SWAVE": 1, "MONTHCODE": month, "RIN_UNIV": 1,
                         "TAGE_EHC": 40, "ESEX": 1, "EEDUC": 39, "ERACE": 1, "EORIGIN": 2,
                         "RMESR": 1, "RMNUMJOBS": 1, "RHLTHMTH": 1, "EMS_EHC": 6,
                         "RHNUMPER": 2, "RHNUMU18": 1, "TFINCPOV": 1.25, "RFPOV": 1000,
                         "TPEARN": month * 100, "TMWKHRS": month})
    frame = pd.DataFrame(rows)
    no_job = frame.PNUM.eq("101")
    frame.loc[no_job, ["TPEARN", "TMWKHRS"]] = np.nan
    frame.loc[no_job, ["RMNUMJOBS", "RMESR"]] = [0, 8]
    frame.loc[frame.PNUM.eq("102") & frame.MONTHCODE.eq(1), "TPEARN"] = -100
    frame.loc[frame.PNUM.eq("102") & frame.MONTHCODE.eq(3), ["RMESR", "RHLTHMTH", "EMS_EHC", "TFINCPOV"]] = [4, 2, 4, 0]
    frame.loc[frame.PNUM.eq("103") & frame.MONTHCODE.eq(1), "TMWKHRS"] = np.nan
    frame.loc[frame.PNUM.eq("104") & frame.MONTHCODE.eq(6), "TFINCPOV"] = np.nan
    frame.loc[frame.PNUM.eq("105") & frame.MONTHCODE.eq(8), "TAGE_EHC"] = 65
    return frame


def run(frame, tmp_path):
    source = tmp_path / "source.csv"
    frame.to_csv(source, sep="|", index=False)
    return prepare_panel(source, tmp_path / "prepared", chunksize=7)


def test_quarterly_values_selection_and_audit(tmp_path):
    metadata = run(fixture(), tmp_path)
    wide = pd.read_csv(tmp_path / "prepared" / "wide.csv")
    long = pd.read_csv(tmp_path / "prepared" / "long.csv")
    schema = json.loads((__import__("pathlib").Path(__file__).parents[1] / "replication/model_export/multivariable/schema.json").read_text())
    assert list(wide.columns) == [c["name"] for c in schema["columns"]]
    assert metadata["retained_persons"] == 2
    assert wide.loc[0, "earnings_q1"] == wide.loc[0, "hours_q1"] == 0
    assert wide.loc[1, "earnings_q1"] == 400
    assert wide.loc[1, "hours_q1"] == 2
    assert wide.loc[1, "employment_q1"] == 4
    assert wide.loc[1, "insurance_q1"] == 2
    assert wide.loc[1, "marital_q1"] == 4
    assert wide.loc[1, "poverty_ratio_q1"] == 0
    assert metadata["retained_negative_earnings_person_months"] == 1
    assert metadata["structural_zero_fills"]["TPEARN"]["retained_person_months"] == 12
    assert long.groupby("person_id").quarter.apply(list).tolist() == [[1, 2, 3, 4]] * 2
    assert len(long.household_id.unique()) == 1
    assert sum(row["excluded_persons"] for row in metadata["sample_flow"]) == 3
    assert metadata["missingness_before_selection"]["poverty_ratio"]["affected_persons"] == 1


def test_disagreement_never_fabricates_zero_and_observed_loss_retained(tmp_path):
    frame = fixture().loc[lambda x: x.PNUM.isin(["101", "102"])].copy()
    frame.loc[frame.PNUM.eq("101") & frame.MONTHCODE.eq(1), "RMESR"] = 1
    metadata = run(frame, tmp_path)
    assert metadata["retained_persons"] == 1
    assert metadata["no_job_recode_disagreement_person_months"] == 1
    assert pd.read_csv(tmp_path / "prepared" / "wide.csv").iloc[0].earnings_q1 == 400


def test_quarter_end_validity_and_december_education(tmp_path):
    frame = fixture().loc[lambda x: x.PNUM.isin(["101", "102"])].copy()
    frame.loc[frame.PNUM.eq("102") & frame.MONTHCODE.eq(1), ["RHLTHMTH", "TFINCPOV", "EEDUC"]] = [np.nan, np.nan, np.nan]
    frame.loc[frame.PNUM.eq("102") & frame.MONTHCODE.eq(12), "EEDUC"] = 46
    metadata = run(frame, tmp_path)
    assert metadata["retained_persons"] == 2
    assert pd.read_csv(tmp_path / "prepared" / "wide.csv").loc[1, "education"] == 46


@pytest.mark.parametrize("change", ["duplicate", "wave", "month", "nonconstant", "children", "poverty_universe"])
def test_invalid_inputs_and_complete_case_rules(tmp_path, change):
    frame = fixture().loc[lambda x: x.PNUM.eq("102")].copy()
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        expected = "Duplicate"
    elif change == "wave":
        frame.loc[frame.index[0], "SWAVE"] = 2
        expected = "wave 1"
    elif change == "month":
        frame.loc[frame.index[0], "MONTHCODE"] = 13
        expected = "MONTHCODE"
    else:
        if change == "nonconstant":
            frame.loc[frame.MONTHCODE.eq(5), "ERACE"] = 2
        elif change == "children":
            frame.loc[frame.MONTHCODE.eq(3), "RHNUMU18"] = 3
        else:
            frame.loc[frame.MONTHCODE.eq(3), "RFPOV"] = np.nan
        expected = "No people satisfy"
    with pytest.raises(ValueError, match=expected):
        run(frame, tmp_path)
