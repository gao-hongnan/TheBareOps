from unittest import mock

import pandas as pd
import pytest
from metadata.core import Metadata
from pipeline_dataops.transform.core import cast_columns


@pytest.mark.skip(
    reason="""
    Not implemented yet.
    Given a raw dataframe, perfrom transformations on it and
    convert it to a transformed dataframe and compare it with
    the expected transformed dataframe.

    In my current transform.py, there is really only one
    function cast_columns, so the point of this test might
    be slightly moot. But consider that there are a bunch
    of transformations that we might want to do in the
    future, e.g. one-hot encoding, etc. Then this test
    will serve as a good example of how to write an
    integration test for a transformation function.
    """
)
def test_transform(sample_raw_df, sample_transformed_df):
    assert True
