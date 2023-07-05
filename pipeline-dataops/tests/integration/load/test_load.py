from unittest import mock

import pandas as pd
import pytest

from metadata.core import Metadata
from pipeline_dataops.load.core import to_bigquery, to_google_cloud_storage


@pytest.mark.skip(
    reason="Not implemented yet. Load a raw dataframe to a staging table in BigQuery."
)
def test_load_to_bigquery(sample_raw_df):
    assert True


@pytest.mark.skip(
    reason="""
    Not implemented yet.
    Load a raw dataframe to a folder in Google Cloud Storage.
    Note that the folder name is `updated_at`, see my `load`
    method in `pipeline.py`.
    """
)
def test_load_to_google_cloud_storage(sample_raw_df):
    assert True
