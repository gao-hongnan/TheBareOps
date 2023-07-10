from typing import List, Union

from pydantic import BaseModel, Field


class CreateBaselineModel(BaseModel):
    name: str = "sklearn.dummy.DummyClassifier"
    strategy: str = "prior"


class CreateModel(BaseModel):
    name: str = "sklearn.linear_model.SGDClassifier"
    loss: str = "log_loss"
    penalty: str = "l2"
    alpha: float = 0.0001
    max_iter: int = 100
    learning_rate: str = "optimal"
    eta0: float = 0.1
    power_t: float = 0.1
    warm_start: bool = True
    random_state: int = 1992


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
    create_baseline_model: CreateBaselineModel = Field(default=CreateBaselineModel())
    create_model: CreateModel = Field(default=CreateModel())

    create_imputer: CreateImputer = Field(default=CreateImputer())
    create_encoder: CreateEncoder = Field(default=CreateEncoder())
    create_standard_scaler: CreateStandardScaler = Field(default=CreateStandardScaler())
    features: Features = Field(default=Features())
