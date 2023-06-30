from unittest import mock
import pytest
from pipeline_dataops.extract.core import (
    from_api,
    interval_to_milliseconds,
    get_url,
    prepare_params,
)
from metadata.core import Metadata
import pandas as pd


@pytest.fixture
def mocked_requests():
    with mock.patch("requests.get") as mock_get:
        yield mock_get


# def test_from_api(sample_raw_df, mocked_requests):
#     # Mock the API response
#     response = mock.Mock()
#     mocked_requests.return_value = response
#     response.json.return_value = {"klines": sample_raw_df.values.tolist()}

#     # Set up the inputs to the function
#     response_columns = [
#         "open_time",
#         "open",
#         "high",
#         "low",
#         "close",
#         "volume",
#         "close_time",
#         "quote_asset_volume",
#         "number_of_trades",
#         "taker_buy_base_asset_volume",
#         "taker_buy_quote_asset_volume",
#         "ignore",
#     ]
#     symbol = "BTCUSDT"
#     start_time = 1624393200000
#     end_time = 1624396800000
#     interval = "1m"
#     limit = 1000
#     base_url = "https://api.binance.com"
#     endpoint = "/api/v3/klines"

#     metadata = Metadata()

#     # Call the function with the inputs
#     metadata = from_api(
#         metadata,
#         response_columns,
#         symbol,
#         start_time,
#         end_time=end_time,
#         interval=interval,
#         limit=limit,
#         base_url=base_url,
#         endpoint=endpoint,
#     )

#     print(metadata.raw_df)

#     # Check if the DataFrame in the returned Metadata object is as expected
#     assert metadata.raw_df.equals(sample_raw_df)

#     # You can also check that the request was made with the correct parameters
#     url = get_url(base_url, endpoint)
#     params = prepare_params(symbol, interval, start_time, limit)
#     mocked_requests.assert_called_once_with(url, params=params)
