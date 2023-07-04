from typing import List, Literal

from pydantic import BaseModel, Field


class LoadToLocal(BaseModel):
    """
    Load Processed Data to Local Machine.

    Attributes
    ----------
    query : str
        SQL query to fetch data from the database.
    """

    output_filename: str = Field(
        default="processed_binance_btcusdt_spot",
        description="Output filename.",
    )


class CastColumns(BaseModel):
    column_types: List[str] = Field(
        default={
            "utc_datetime": "datetime64[ns]",
            "open_time": "datetime64[ns]",
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
            "close_time": "datetime64[ns]",
            "quote_asset_volume": "float",
            "number_of_trades": "int",
            "taker_buy_base_asset_volume": "float",
            "taker_buy_quote_asset_volume": "float",
            "ignore": "str",
            "updated_at": "datetime64[ns]",
        }
    )


class Preprocess(BaseModel):
    load_to_local: LoadToLocal = Field(
        default=LoadToLocal(), description="Load Processed Data to Local Machine."
    )
    cast_columns: CastColumns = Field(default=CastColumns(), description="Cast Columns")
