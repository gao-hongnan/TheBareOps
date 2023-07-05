from unittest import mock

import pytest
import requests

from pipeline_dataops.extract.core import (
    execute_request,
    get_url,
    handle_response,
    interval_to_milliseconds,
    prepare_params,
)


@pytest.mark.skip(reason="Not implemented yet")
def test_interval_to_milliseconds():
    assert True


@pytest.mark.skip(reason="Not implemented yet")
def test_get_url():
    assert True


@pytest.mark.skip(reason="Not implemented yet")
def test_handle_response():
    assert True


@pytest.mark.skip(reason="Not implemented yet")
def test_prepare_params():
    assert True


@pytest.mark.skip(
    reason=""""
    Not implemented yet.
    This test can potentially be an integration test since it requires
    both url and params, which depend on get_url and prepare_params.
    """
)
def test_execute_request():
    # Mock the requests.get function
    with mock.patch.object(requests, "get") as mock_get:
        # Create a mock response object with a .json() method
        mock_response = mock.Mock()
        mock_response.json.return_value = [{"key": "value"}]

        # Set the return value of requests.get() to the mock response
        mock_get.return_value = mock_response

        # Call execute_request, which will use the mocked requests.get()
        params = {"key": "value"}
        url = "http://test.url"
        result = execute_request(url, params)

        # Assert that the function returned the expected result
        assert result == [{"key": "value"}]

        # Assert requests.get was called with the right arguments
        mock_get.assert_called_once_with(url=url, params=params, timeout=120)
