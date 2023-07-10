from typing import Dict, List, Literal, Union, Optional

from pydantic import BaseModel, Field
from pydantic import BaseModel
from typing import Any, Dict, List, Union, Optional
from sklearn.base import BaseEstimator


class BaseModelConfig(BaseModel):
    name: str
    random_state: Optional[int] = 42


class DummyModelConfig(BaseModelConfig):
    strategy: str


class LogisticModelConfig(BaseModelConfig):
    solver: str


class RandomForestModelConfig(BaseModelConfig):
    n_estimators: Optional[int] = 100


class CreateImputer(BaseModel):
    name: str = "KNNImputer"
    n_neighbors: int = 5
    weights: str = "uniform"


class CreateEncoder(BaseModel):
    name: str = "OneHotEncoder"
    handle_unknown: str = "ignore"


class CreateStandardScaler(BaseModel):
    name: str = "StandardScaler"
    with_mean: bool = True
    with_std: bool = True


class Features(BaseModel):
    continuous_features: List[str] = Field(
        default=[
            "open",
            "high",
            "low",
            "volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ],
        description="Continuous columns.",
    )

    categorical_features: List[str] = Field(
        default=[], description="Categorical columns."
    )

    target_columns: Union[str, List[str]] = Field(
        default="price_increase", description="Target column."
    )


class Train(BaseModel):
    # model_configs: Dict[str, BaseModelConfig] = Field(
    #     default={
    #         "DummyClassifier": {
    #             "model_name": "DummyClassifier",
    #             "strategy": "most_frequent",
    #         },
    #     }
    # )

    create_imputer: CreateImputer = Field(default=CreateImputer())
    create_encoder: CreateEncoder = Field(default=CreateEncoder())
    create_standard_scaler: CreateStandardScaler = Field(default=CreateStandardScaler())
    features: Features = Field(default=Features())
