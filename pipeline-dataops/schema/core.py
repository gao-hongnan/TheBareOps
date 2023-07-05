from datetime import datetime
from typing import List

import pandas as pd
from google.cloud import bigquery

from schema.base import BaseSchema


class RawSchema(BaseSchema):
    utc_datetime: datetime
    open_time: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    close_time: int
    quote_asset_volume: str
    number_of_trades: int
    taker_buy_base_asset_volume: str
    taker_buy_quote_asset_volume: str
    ignore: str
    updated_at: datetime


# Then define TransformedSchema with its unique fields


class TransformedSchema(BaseSchema):
    utc_datetime: datetime
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float
    ignore: str
    updated_at: datetime
    utc_singapore_datetime: datetime


# TODO: Consider adding schema under a pydantic model
# since it can do validation and is generally used for
# schema creation + validation. Furthermore, creating
# schema from pandas is not robust.
def generate_bq_schema_from_pandas(df: pd.DataFrame) -> List[bigquery.SchemaField]:
    """
    Convert pandas dtypes to BigQuery dtypes.

    Parameters
    ----------
    dtypes : pandas Series
        The pandas dtypes to convert.

    Returns
    -------
    List[google.cloud.bigquery.SchemaField]
        The corresponding BigQuery dtypes.
    """
    dtype_mapping = {
        "int64": bigquery.enums.SqlTypeNames.INT64,
        "float64": bigquery.enums.SqlTypeNames.FLOAT64,
        "object": bigquery.enums.SqlTypeNames.STRING,
        "bool": bigquery.enums.SqlTypeNames.BOOL,
        "datetime64[ns]": bigquery.enums.SqlTypeNames.DATETIME,
        "datetime64[ns, Asia/Singapore]": bigquery.enums.SqlTypeNames.DATETIME,
    }

    schema = []

    for column, dtype in df.dtypes.items():
        if str(dtype) not in dtype_mapping:
            raise ValueError(f"Cannot convert {dtype} to a BigQuery data type.")

        bq_dtype = dtype_mapping[str(dtype)]
        field = bigquery.SchemaField(name=column, field_type=bq_dtype, mode="NULLABLE")
        schema.append(field)

    return schema
