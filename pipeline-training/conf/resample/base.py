"""
This module contains Pydantic models for specifying the configuration for data
resampling strategies and their parameters, especially focused on splitting
the data into training, validation, and test sets.

The module currently supports a simple train-test split configuration but is
designed to be extensible to other resampling strategies, such as
K-Fold cross-validation, Stratified splits, etc.

Classes
-------
TrainTestSplit
    A Pydantic model for specifying the configuration for the train-test split
    strategy.
Resample
    A Pydantic model for specifying the resampling strategy and its associated
    configuration.

TODO
----
1. Add support for other resampling strategies, especially K-Folds and its
   variants.
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class TrainTestSplit(BaseModel):
    """Configuration for splitting data into training and test sets.

    Attributes
    ----------
    initial_split : Dict[str, Any]
        Configuration for the initial split of the data into training
        and [test, validation]. Defaults to a 70/30 split.
    secondary_split : Dict[str, Any]
        Configuration for the secondary split of the data into test
        and validation sets. Defaults to a 50/50 split.
    """

    initial_split: Dict[str, Any] = Field(
        default={
            "train_size": 0.7,
            "random_state": 1992,
            "shuffle": True,
        },
        description="Initial split of the data: train and [test, val].",
    )
    secondary_split: Dict[str, Any] = Field(
        default={
            "train_size": 0.5,
            "random_state": 1992,
            "shuffle": True,
        },
        description="Secondary split of the data: test and val.",
    )


class Resample(BaseModel):
    """Configuration for resampling strategies.

    Attributes
    ----------
    strategy_name : str
        Name of the resampling strategy. Currently supports "train_test_split".
    strategy_config : TrainTestSplit
        Configuration for the chosen resampling strategy. Defaults to
        the TrainTestSplit config.
    """

    strategy_name: str = Field(default="train_test_split")
    strategy_config: TrainTestSplit = Field(default=TrainTestSplit())
