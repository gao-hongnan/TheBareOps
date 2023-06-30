from typing import List

import pandas as pd


# TODO: god help me the tz localize and convert is not working
def cast_columns(
    df: pd.DataFrame,
    datetime_column: str,
    timezone_column: str,
    timezone: str,
    int_time_columns: List[str],
    float_columns: List[str],
) -> pd.DataFrame:
    """
    Cast specified columns of the input DataFrame to appropriate data types.

    This function converts the values in a datetime column from Unix time
    (milliseconds since 1970-01-01) to a datetime format. It also creates
    a new column with the localized time, converts specified integer
    timestamp columns to datetime format, and converts specified columns
    to float data type.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame whose columns are to be cast.
    datetime_column : str
        Column in DataFrame with Unix timestamps to convert to datetime.
    timezone_column : str
        Column to be created in DataFrame for storing localized time.
    timezone : str
        Timezone to which datetime_column will be converted.
    int_time_columns : List[str]
        List of columns in DataFrame with integer timestamps to convert
        to datetime.
    float_columns : List[str]
        List of columns in DataFrame to convert to float data type.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with specified columns cast to appropriate types.
    """
    df = df.copy()

    # TODO: Consider adding this to great expectations because the datetime
    # column is not being recognized as a datetime column.
    df[datetime_column] = pd.to_datetime(df[datetime_column], unit="ms")

    df[timezone_column] = (
        df[datetime_column].dt.tz_localize("UTC").dt.tz_convert(timezone)
    )

    for int_time_col in int_time_columns:
        df[int_time_col] = pd.to_datetime(df[int_time_col], unit="ms")

    for float_col in float_columns:
        df[float_col] = df[float_col].astype(float)

    return df
