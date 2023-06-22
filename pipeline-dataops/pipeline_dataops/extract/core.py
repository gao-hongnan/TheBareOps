import math
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
from mlops_pipeline_feature_v1.utils import interval_to_milliseconds


def from_api(
    symbol: str,
    start_time: int,
    end_time: Optional[int] = None,
    interval: str = "1m",
    limit: int = 1000,
    base_url: str = "https://api.binance.com",
    endpoint: str = "/api/v3/klines",
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    url = base_url + endpoint

    # Convert interval to milliseconds
    interval_in_milliseconds = interval_to_milliseconds(interval)

    time_range = end_time - start_time  # total time range
    request_max = limit * interval_in_milliseconds

    start_iteration = start_time
    end_iteration = start_time + request_max

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
    }

    if end_time is not None:
        params["endTime"] = end_time

    response_columns = [
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

    if time_range <= request_max:  # time range selected within 1000 rows limit
        resp = requests.get(url=url, params=params, timeout=30)
        data = resp.json()
        df = pd.DataFrame(data, columns=response_columns)
        time.sleep(1)
    elif (
        time_range > request_max
    ):  # start_time and end_time selected > limit rows of data
        df = pd.DataFrame()  # empty dataframe to append to
        num_iterations = math.ceil(time_range / request_max)  # number of loops required

        for _ in range(num_iterations):
            # make request with updated params
            resp = requests.get(url=url, params=params, timeout=30)
            data = resp.json()
            _df = pd.DataFrame(data, columns=response_columns)

            df = pd.concat([df, _df])

            start_iteration = end_iteration
            end_iteration = min(
                end_iteration + request_max, end_time
            )  # don't go beyond the actual end time
            # adjust params

            params["startTime"], params["endTime"] = (
                start_iteration,
                end_iteration,
            )  # adjust params
            time.sleep(1)

    df.insert(0, "utc_datetime", pd.to_datetime(df["open_time"], unit="ms"))

    # prepare metadata
    metadata = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time,
        "base_url": base_url,
        "endpoint": endpoint,
        "exported_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    return df, metadata
