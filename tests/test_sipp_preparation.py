import numpy as np
import pandas as pd
import pytest

from replication.model_export.prepare_sipp import prepare_panel


def monthly_fixture():
    rows = []
    for person in (101, 102, 103):
        for month in range(1, 13):
            rows.append({"SSUID": "00000000000001", "PNUM": str(person),
                         "SPANEL": 2018, "SWAVE": 1, "MONTHCODE": month,
                         "RIN_UNIV": 1, "TAGE": 40, "RMNUMJOBS": 1,
                         "TPEARN": float(month * 100), "TMWKHRS": float(month)})
    frame = pd.DataFrame(rows)
    no_job = frame.PNUM.eq("101")
    frame.loc[no_job, ["TPEARN", "TMWKHRS"]] = np.nan
    frame.loc[no_job, "RMNUMJOBS"] = 0
    frame.loc[frame.PNUM.eq("102") & frame.MONTHCODE.eq(1), "TPEARN"] = -100
    frame.loc[frame.PNUM.eq("103") & frame.MONTHCODE.eq(1), "TMWKHRS"] = np.nan
    return frame


def test_structural_zeros_business_losses_and_quarterly_units(tmp_path):
    source = tmp_path / "input.csv"
    monthly_fixture().to_csv(source, sep="|", index=False)
    output = tmp_path / "prepared"
    metadata = prepare_panel(source, output, chunksize=7)
    panel = pd.read_csv(output / "wide.csv")
    assert metadata["retained_persons"] == 2
    assert metadata["sample_flow"][-1]["excluded_persons"] == 1
    assert metadata["structural_zero_fills"]["hours"]["retained_person_months"] == 12
    np.testing.assert_array_equal(panel.iloc[0], np.zeros(8))
    np.testing.assert_array_equal(panel.iloc[1], [400, 1500, 2400, 3300, 2, 5, 8, 11])
    long = pd.read_csv(output / "long.csv")
    assert long.groupby("person_id").quarter.apply(list).tolist() == [[1, 2, 3, 4]] * 2
    assert metadata["retained_negative_earnings_person_months"] == 1


@pytest.mark.parametrize("invalid", ["duplicate", "wrong_wave", "partial_year"])
def test_bad_panel_structure_is_not_silently_pooled(tmp_path, invalid):
    frame = monthly_fixture()
    if invalid == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        expected = "duplicate person-month"
    elif invalid == "wrong_wave":
        frame.loc[0, "SWAVE"] = 2
        expected = "wave 1"
    else:
        frame = frame.loc[frame.MONTHCODE.ne(12)]
        expected = "No people satisfy"
    source = tmp_path / "input.csv"
    frame.to_csv(source, sep="|", index=False)
    with pytest.raises(ValueError, match=expected):
        prepare_panel(source, tmp_path / "prepared")
