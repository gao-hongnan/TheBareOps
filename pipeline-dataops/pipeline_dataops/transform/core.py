import pandas as pd
from typing import List


# TODO: god help me the tz localize and convert is not working
def cast_columns(
    df: pd.DataFrame,
    datetime_column: str,
    timezone_column: str,
    timezone: str,
    int_time_columns: List[str],
    float_columns: List[str],
) -> pd.DataFrame:
    df = df.copy()

    df[timezone_column] = (
        df[datetime_column].dt.tz_localize(timezone).dt.tz_convert(timezone)
    )

    for int_time_col in int_time_columns:
        df[int_time_col] = pd.to_datetime(df[int_time_col], unit="ms")

    for float_col in float_columns:
        df[float_col] = df[float_col].astype(float)

    return df
