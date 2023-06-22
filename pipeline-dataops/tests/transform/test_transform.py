import pytest
from pipeline_dataops.transform.core import cast_columns
import pandas as pd

# Parameters for cast_columns function
datetime_column = "utc_datetime"
timezone_column = "utc_singapore_time"
timezone = "Asia/Singapore"
int_time_columns = ["open_time"]
float_columns = ["open", "close", "volume"]


@pytest.mark.parametrize(
    "datetime_column, timezone_column, timezone, int_time_columns, float_columns",
    [(datetime_column, timezone_column, timezone, int_time_columns, float_columns)],
)
def test_cast_columns(
    sample_df,
    datetime_column,
    timezone_column,
    timezone,
    int_time_columns,
    float_columns,
):
    result_df = cast_columns(
        sample_df,
        datetime_column,
        timezone_column,
        timezone,
        int_time_columns,
        float_columns,
    )
    # check timezone conversion
    assert (
        result_df[timezone_column]
        == sample_df[datetime_column].dt.tz_localize("UTC").dt.tz_convert(timezone)
    ).all()

    # check integer to datetime conversion
    for col in int_time_columns:
        assert (result_df[col] == pd.to_datetime(sample_df[col], unit="ms")).all()

    # check float conversion
    for col in float_columns:
        assert (result_df[col] == sample_df[col].astype(float)).all()
