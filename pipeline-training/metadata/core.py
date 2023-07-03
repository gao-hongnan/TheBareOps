from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Union

import pandas as pd
import pytz

# TODO: Compose Metadata in a similar way as Config instead of scattering.
# TODO: Add more metadata attributes, for example the GCS path that stores
# raw and transformed data.


# pylint: disable=unnecessary-dunder-call
# mutable
@dataclass(frozen=False)
class Metadata:
    # tracks the inner state of the pipeline, update as it traverses the pipeline.
    # e.g. if the pipeline fails at a certain stage, we can use this to restart
    # from that stage.

    # general
    pipeline_name: str = None
    git_commit_hash: str = None

    # inside extract.py
    raw_df: pd.DataFrame = None
    raw_num_rows: int = None
    raw_num_cols: int = None
    raw_dataset: str = None
    raw_table_name: str = None
    raw_query: str = None

    # validate_raw
    raw_validation_dict: Dict[str, Any] = None

    # inside load.py
    raw_file_size: int = None
    raw_file_format: str = None
    raw_dvc_metadata: Dict[str, Any] = None

    # inside transform.py
    processed_df: pd.DataFrame = None
    processed_num_rows: int = None
    processed_num_cols: int = None
    processed_file_size: int = None
    processed_dvc_metadata: Dict[str, Any] = None

    # inside resampling.py
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    X_val: pd.DataFrame = None
    y_train: pd.DataFrame = None
    y_test: pd.DataFrame = None
    y_val: pd.DataFrame = None

    # inside train.py
    model_artifacts: Dict[str, Any] = None

    # inside evaluate.py
    best_params: Dict[str, Any] = None

    def release(self, attribute: str) -> Any:
        self.__setattr__(attribute, None)

    def set_attrs(self, attr_dict: Dict[str, Any]):
        """Sets attributes on the Metadata instance.

        Parameters
        ----------
        attr_dict: Dict[str, Any]
            A dictionary where keys are attribute names and values are the
            corresponding values to be set.
        """
        for attr_name, value in attr_dict.items():
            self.__setattr__(attr_name, value)
