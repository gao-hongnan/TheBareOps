from datetime import datetime
from typing import List, Optional

import pytz
from pydantic import BaseModel, Field
from rich.pretty import pprint


class FromAPI(BaseModel):
    symbol: str = Field(default="BTCUSDT")
    start_time: int = Field(default=1687363200000)
    end_time: Optional[int] = Field(
        default_factory=lambda: int(
            datetime.now(pytz.timezone("Asia/Singapore")).timestamp() * 1000
        )
    )
    interval: str = Field(default="1m")
    limit: int = Field(default=1000)
    base_url: str = Field(default="https://api.binance.com")
    endpoint: str = Field(default="/api/v3/klines")
    response_columns: List[str] = Field(
        default=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
    )

    class Config:
        frozen = False  # mutable because I need to overwrite start_time and end_time


class Extract(BaseModel):
    from_api: FromAPI = Field(default=FromAPI())


if __name__ == "__main__":
    extract = Extract()
    pprint(extract)
