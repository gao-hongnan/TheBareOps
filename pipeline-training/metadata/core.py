from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn import pipeline

# TODO: Compose Metadata in a similar way as Config instead of scattering.
# TODO: Add more metadata attributes, for example the GCS path that stores
# raw and transformed data.


# pylint: disable=unnecessary-dunder-call
# mutable
@dataclass(frozen=False)
class Metadata:
    """Tracks the inner state of the pipeline, update as it traverses the pipeline."""

    # e.g. if the pipeline fails at a certain stage, we can use this to restart
    # from that stage.

    # inside extract.py
    raw_df: pd.DataFrame = None
    raw_num_rows: int = None
    raw_num_cols: int = None
    raw_dataset: str = None
    raw_table_name: str = None
    raw_query: str = None

    # inside load.py
    raw_file_size: int = None
    raw_file_format: str = None
    raw_filepath: str = None
    raw_dvc_metadata: Dict[str, Any] = None

    # inside clean.py
    processed_df: pd.DataFrame = None
    processed_num_rows: int = None
    processed_num_cols: int = None
    processed_file_size: int = None
    processed_file_format: str = None
    processed_dvc_metadata: Dict[str, Any] = None
    X: Union[pd.DataFrame, np.ndarray] = None
    y: Union[pd.DataFrame, np.ndarray] = None
    feature_columns: List[str] = None
    target_columns: Union[str, List[str]] = None
    cleaned_num_cols: int = None

    # inside resample.py
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    X_val: pd.DataFrame = None
    y_train: pd.DataFrame = None
    y_test: pd.DataFrame = None
    y_val: pd.DataFrame = None

    # model_training/preprocess.py
    numeric_features: List[str] = None
    categorical_features: List[str] = None
    preprocessor: pipeline.Pipeline = None
    X_train: np.ndarray = None
    X_val: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_val: np.ndarray = None
    y_test: np.ndarray = None
    num_classes: int = None
    classes: List[int] = None

    # inside train.py
    model_artifacts: Dict[str, Any] = None
    run_id: str = None
    model_version: str = None
    experiment_id: str = None
    experiment_name: str = None
    artifact_uri: str = None

    # inside optimize.py
    # trials is trial number to trial dict
    trials: Dict[int, Dict[str, Any]] = None
    trials_df: pd.DataFrame = None
    best_params: Dict[str, Any] = None

    # inside evaluate.py
    holdout_performance: Dict[str, float] = None
    avg_expected_loss: float = None
    avg_bias: float = None
    avg_variance: float = None

    def release(self, attribute: str) -> Any:
        """Releases an attribute from the Metadata instance."""
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
