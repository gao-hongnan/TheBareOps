from typing import Dict, List, Literal, Union

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


class ExtractFeaturesAndTarget(BaseModel):
    feature_columns: List[str] = Field(
        default=[
            "open",
            "high",
            "low",
            "volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ],
        description="Feature columns.",
    )
    target_columns: Union[str, List[str]] = Field(
        default="price_increase", description="Target column."
    )
    as_dataframe: Literal[True, False] = Field(
        default=True, description="Return as dataframe."
    )


class CastColumns(BaseModel):
    column_types: Dict[str, str] = Field(
        default={
            "utc_datetime": "datetime64[ns]",
            "open_time": "int",
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
            "close_time": "int",
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
    extract_features_and_target: ExtractFeaturesAndTarget = Field(
        default=ExtractFeaturesAndTarget(as_dataframe=False),
        description="Extract Features and Target",
    )
