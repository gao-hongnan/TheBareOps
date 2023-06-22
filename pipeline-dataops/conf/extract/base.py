from datetime import datetime
from typing import Optional

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


class Extract(BaseModel):
    from_api: FromAPI = Field(default=FromAPI())


if __name__ == "__main__":
    extract = Extract()
    pprint(extract)
