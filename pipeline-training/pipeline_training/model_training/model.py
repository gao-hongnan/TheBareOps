from typing import Any, Dict

from rich.pretty import pprint
from sklearn.base import BaseEstimator

from conf.base import Config
from typing import Any, Dict, List
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from typing import Any, Dict
from conf.training.base import BaseModelConfig

# pylint: disable=eval-used


def create_model(model_config: Dict[str, Any]) -> BaseEstimator:
    model_class = eval(model_config.pop("model_name"))
    model = model_class(**model_config)
    return model


class ModelFactory:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def create_all(self) -> List[BaseEstimator]:
        return [self.create_model(config) for config in self.configs]

    def create_model(self) -> BaseEstimator:
        model_class = eval(model_config.model_name)
        # Convert the model config back to a dictionary and remove 'model_name'
        model_args = model_config.model_dump(mode="python")
        model_args.pop("model_name")
        model = model_class(**model_args)
        return model
