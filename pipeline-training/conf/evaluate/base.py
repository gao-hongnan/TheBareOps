from typing import Any, Dict

from pydantic import BaseModel, Field


class PredictOnHoldoutSet(BaseModel):
    prefix: str = "holdout"


class BiasVariance(BaseModel):
    loss: str = "0-1_loss"
    num_rounds: int = 200
    random_seed: int = 42


class Evaluate(BaseModel):
    predict_on_holdout_set: PredictOnHoldoutSet = PredictOnHoldoutSet()
    bias_variance: BiasVariance = BiasVariance()
