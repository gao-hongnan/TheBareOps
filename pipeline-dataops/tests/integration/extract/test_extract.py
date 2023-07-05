from unittest import mock

import pandas as pd
import pytest

from metadata.core import Metadata
from pipeline_dataops.extract.core import from_api


@pytest.fixture
def mocked_requests():
    with mock.patch("requests.get") as mock_get:
        yield mock_get


@pytest.mark.skip(
    reason="""
    Not implemented yet.
    This test, 'test_from_api', is an integration test because it tests
    the interaction between the 'from_api' function and the actual API endpoint.
    It checks if the function correctly handles responses from the API and
    properly processes the raw data. Therefore, it involves the integration
    of multiple components of the system, and not just the behavior of
    individual units in isolation.
    """
)
def test_from_api(sample_raw_df):
    assert True
