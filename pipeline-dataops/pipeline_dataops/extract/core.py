import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger
from rich.pretty import pprint

from conf.base import Config
from metadata.core import Metadata

# pylint: disable=invalid-name


def interval_to_milliseconds(interval: str) -> int:
    """
    Convert a time interval to milliseconds.

    This function takes as input a string representing a time interval
    and converts it into milliseconds. The input string should end with
    'm' (for minutes), 'h' (for hours), or 'd' (for days).

    Parameters
    ----------
    interval : str
        A string representing the time interval. It should be a number
        followed by a single character: 'm' (minutes), 'h' (hours), or 'd'
        (days). For example, '1m' stands for 1 minute, '2h' stands for
        2 hours, and '3d' stands for 3 days.

    Returns
    -------
    int
        The time interval converted into milliseconds.

    Raises
    ------
    ValueError
        If the interval does not end with 'm', 'h', or 'd'.
    """
    if interval.endswith("m"):
        return int(interval[:-1]) * 60 * 1000
    if interval.endswith("h"):
        return int(interval[:-1]) * 60 * 60 * 1000
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60 * 60 * 1000
    raise ValueError(f"Invalid interval format: {interval}")


def get_url(base_url: str, endpoint: str) -> str:
    return base_url + endpoint


def prepare_params(
    symbol: str,
    interval: str,
    start_time: int,
    limit: int,
    end_time: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepares the parameters for the API request.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g., 'BTCUSDT').
    interval : str
        Interval for the data (e.g., '1m' for 1 minute).
    start_time : int
        Start time for data collection in milliseconds since epoch.
    limit : int
        Limit on the number of data points to retrieve.
    end_time : int, optional
        End time for data collection in milliseconds since epoch.
        If not provided, the current time is used.

    Returns
    -------
    params : Dict[str, Any]
        Dictionary containing the prepared parameters for the API request.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
    }

    if end_time is not None:
        params["endTime"] = end_time

    return params


def execute_request(url: str, params: Dict[str, Any]) -> List[Union[str, float]]:
    """
    Sends a GET request to a specified URL with given parameters and
    returns the response data.

    This function sends a GET request to the specified URL with the
    provided parameters. The server's response is expected to be a
    JSON object. This function will convert the JSON response into a
    Python object using the .json() method from the requests library.
    It is expected that the JSON response from the server is an array
    of JSON objects, as the return type is a list.

    Parameters
    ----------
    url : str
        The URL to send the GET request to.
    params : Dict[str, Any]
        The parameters to include in the GET request.

    Returns
    -------
    data: List[Union[str, float]]
        The data returned from the GET request, a list containing either
        string or float elements.

    Raises
    ------
    requests.exceptions.RequestException
        If the GET request fails for any reason.
    """
    resp = requests.get(url=url, params=params, timeout=30)
    data = resp.json()
    return data


# Create a function to handle the response and convert it into a DataFrame.
def handle_response(data: list, response_columns: list):
    df = pd.DataFrame(data, columns=response_columns)
    return df


def from_api(
    metadata: Metadata,
    response_columns: List[str],
    symbol: str,
    start_time: int,
    end_time: Optional[int] = None,
    interval: str = "1m",
    limit: int = 1000,
    base_url: str = "https://api.binance.com",
    endpoint: str = "/api/v3/klines",
) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    url = get_url(base_url, endpoint)

    # Convert interval to milliseconds
    interval_in_milliseconds = interval_to_milliseconds(interval)

    time_range = end_time - start_time  # total time range
    request_max = limit * interval_in_milliseconds

    start_iteration = start_time
    end_iteration = start_time + request_max

    params = prepare_params(symbol, interval, start_time, limit, end_time)

    if time_range <= request_max:
        # NOTE: This chunk means we can retrieve data in one single request
        # where time range selected within 1000 rows limit
        data: List[List[Union[str, float]]] = execute_request(url=url, params=params)
        time.sleep(1)
    elif (
        time_range > request_max
    ):  # start_time and end_time selected > limit rows of data
        num_iterations = math.ceil(time_range / request_max)  # number of loops required
        data = []
        for _ in range(num_iterations):
            # make request with updated params
            _data: List[List[Union[str, float]]] = execute_request(
                url=url, params=params
            )
            data.extend(_data)
            start_iteration = end_iteration
            end_iteration = min(
                end_iteration + request_max, end_time
            )  # don't go beyond the actual end time

            # adjust params
            params["startTime"], params["endTime"] = (
                start_iteration,
                end_iteration,
            )
            time.sleep(1)

    df = handle_response(data, response_columns)
    df.insert(0, "utc_datetime", pd.to_datetime(df["open_time"], unit="ms"))
    updated_at = metadata.updated_at
    df["updated_at"] = updated_at

    # TODO: overwrite the existing data to include the new data, a bit unclean.
    data: List[List[Union[str, float, datetime]]] = df.values.tolist()

    # prepare metadata
    metadata_dict = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time,
        "base_url": base_url,
        "endpoint": endpoint,
        "extract_updated_at_utc": updated_at,
        "raw_data": data,
        "raw_df": df,
    }
    metadata.set_attrs(metadata_dict)
    return metadata


if __name__ == "__main__":
    cfg = Config()
    # pprint(cfg)
    seed_all(cfg.general.seed)

    metadata = Metadata()
    # pprint(metadata)

    logger = Logger(
        log_file="pipeline_training.log",
        log_root_dir=cfg.dirs.stores.logs,
        module_name=__name__,
        propagate=False,
    ).logger

    gcs = GCS(
        project_id=cfg.env.project_id,
        google_application_credentials=cfg.env.google_application_credentials,
        bucket_name=cfg.env.gcs_bucket_name,
    )

    bq = BigQuery(
        project_id=cfg.env.project_id,
        google_application_credentials=cfg.env.google_application_credentials,
        dataset=cfg.env.bigquery_raw_dataset,
        table_name=cfg.env.bigquery_raw_table_name,
    )
    # Query to find the maximum open_date
    query = f"""
        SELECT MAX(open_time) as max_open_time
        FROM `{bq.table_id}`
        """
    max_date_result: pd.DataFrame = bq.query(query, as_dataframe=True)
    max_open_time: int = max(max_date_result["max_open_time"])

    # now max_open_time is your new start_time
    start_time = max_open_time + interval_to_milliseconds(
        interval=cfg.extract.from_api.interval
    )
    logger.warning("Overwriting `start_time` in the config.")
    # NOTE: We are overwriting the start_time here.
    cfg.extract.from_api.start_time = int(start_time)
    cfg.extract.from_api.start_time = int(1687835000000)
    cfg.extract.from_api.start_time = int(1687700000000)

    metadata = from_api(
        metadata=metadata,
        **cfg.extract.from_api.model_dump(mode="python"),
    )
    # pprint(metadata)
    from conf.schema.core import RawSchema

    data = metadata.raw_data
    pprint(type(data[1][0]))
    df = metadata.raw_df
    # RawSchema.model_validate(data)
    # After extraction
    data_dict_list: List[Dict[str, Any]] = df.to_dict(
        "records"
    )  # Convert DataFrame to list of dictionaries
    # pprint(data_dict_list)
    # Validate each item in the list using RawSchema
    for data in data_dict_list:
        valid_data_list = RawSchema.model_validate(data)
        pprint(valid_data_list)

    schema = RawSchema.to_bq_schema()
    pprint(schema)
