"""
Simple train-val-test split instead of K-Folds and its variants.
As mentioned in GCP blog, the output of data prepartion is
the data *splits* in the prepared format.
"""
from typing import Tuple

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from conf.base import Config
from metadata.core import Metadata

# TODO: Consider adding a class and do generator style.


def get_data_splits(
    cfg: Config, metadata: Metadata, X: np.ndarray, y: np.ndarray
) -> Metadata:
    """Generate balanced data splits.

    NOTE: As emphasized this is not a modelling project. This is a project
    to illustrate end-to-end MLOps. Therefore, we will not be using
    a lot of "SOTA" methods.

    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.
    Returns:
        Tuple: data splits as Numpy arrays.
    """
    # 70-15-15 split

    strategy_func = getattr(sklearn.model_selection, cfg.resample.strategy_name)

    X_train, X_, y_train, y_ = strategy_func(
        X, y, stratify=y, **cfg.resample.strategy_config.initial_split
    )
    X_val, X_test, y_val, y_test = strategy_func(
        X_, y_, stratify=y_, **cfg.resample.strategy_config.secondary_split
    )

    attr_dict = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    metadata.set_attrs(attr_dict=attr_dict)
    return metadata
