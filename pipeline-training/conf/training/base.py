from typing import Dict, List, Literal, Union, Optional

from pydantic import BaseModel, Field
from pydantic import BaseModel
from typing import Any, Dict, List, Union, Optional
from sklearn.base import BaseEstimator


class BaseModelConfig(BaseModel):
    model_name: str
    random_state: Optional[int] = 42


class DummyModelConfig(BaseModelConfig):
    strategy: str


class LogisticModelConfig(BaseModelConfig):
    solver: str


class RandomForestModelConfig(BaseModelConfig):
    n_estimators: Optional[int] = 100


class CreateImputer(BaseModel):
    pass


class CreateEncoder(BaseModel):
    pass


class CreateStandardScaler(BaseModel):
    pass


class Train(BaseModel):
    model_configs: Dict[str, BaseModelConfig] = Field(
        default={
            "DummyClassifier": {
                "model_name": "DummyClassifier",
                "strategy": "most_frequent",
            },
        }
    )
