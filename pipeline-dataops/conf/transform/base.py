from pydantic import BaseModel, Field
from typing import List


class CastColumns(BaseModel):
    datetime_column: str = Field(default="utc_datetime")
    timezone_column: str = Field(default="utc_singapore_datetime")
    timezone: str = Field(default="Asia/Singapore")
    int_time_columns: List[str] = Field(default=["open_time", "close_time"])
    float_columns: List[str] = Field(
        default=[
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
    )


class Transform(BaseModel):
    cast_columns: CastColumns = Field(default=CastColumns())
