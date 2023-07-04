from typing import Any, Dict

from pydantic import BaseModel, Field


class TrainTestSplit(BaseModel):
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


# TODO: add support for K-Folds and its variants.
class Resample(BaseModel):
    strategy_name: str = Field(default="train_test_split")
    strategy_config: TrainTestSplit = Field(default=TrainTestSplit())
