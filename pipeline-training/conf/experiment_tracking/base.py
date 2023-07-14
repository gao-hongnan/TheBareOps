from typing import Any, Dict

from pydantic import BaseModel, Field


class Experiment(BaseModel):
    experiment_name: str = "thebareops_mlops_pipeline"
    tracking_uri: str = None
    start_run: Dict[str, Any] = Field(
        default={
            "run_name": "tuned_thebareops_sgd_5_epochs",
            "nested": True,
            "description": "TheBareOps with sklearn SGDClassifier",
            "tags": {"framework": "sklearn", "type": "classification"},
        },
    )
    log_artifacts: Dict[str, Any] = Field(default={"artifact_path": "stores"})
    register_model: Dict[str, Any] = Field(
        default={
            "model_uri": "runs:/{run_id}/artifacts/model",
            "name": "thebareops_sgd",
        }
    )
    set_signature: Dict[str, Any] = Field(
        default={
            "model_uri": "gs://thebareops/thebareops/artifacts/{experiment_id}/{run_id}/artifacts/registry"
        }
    )
