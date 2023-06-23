# conftest.py is a special file name that pytest recognizes as a place to find fixtures.
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing across multiple test files."""
    df = pd.DataFrame(
        {
            "open_time": [1687478400000, 1687478460000, 1687478520000],
            "open": ["30000.0", "30050.0", "30100.0"],
            "high": ["30100.0", "30150.0", "30200.0"],
            "low": ["29950.0", "30000.0", "30050.0"],
            "close": ["30050.0", "30100.0", "30150.0"],
            "volume": ["10.0", "20.0", "30.0"],
            "close_time": [1687478460000, 1687478520000, 1687478580000],
            "quote_asset_volume": ["300500.0", "602000.0", "904500.0"],
            "number_of_trades": [100, 200, 300],
            "taker_buy_base_asset_volume": ["5.0", "10.0", "15.0"],
            "taker_buy_quote_asset_volume": ["150250.0", "301000.0", "452750.0"],
            "ignore": ["ignore1", "ignore2", "ignore3"],
            "utc_datetime": pd.to_datetime(
                [1687478400000, 1687478460000, 1687478520000], unit="ms"
            ),
        }
    )
    return df
