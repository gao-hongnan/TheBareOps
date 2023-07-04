from unittest import mock

import pytest
import requests
from pipeline_dataops.load.core import to_bigquery, to_google_cloud_storage


@pytest.mark.skip(reason="Not implemented yet. Remember to mock the BigQuery client.")
def test_to_bigquery():
    assert True


@pytest.mark.skip(reason="Not implemented yet. Remember to mock the GCS client.")
def test_to_google_cloud_storage():
    assert True
