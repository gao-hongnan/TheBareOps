import mlflow
import numpy as np
from common_utils.core.logger import Logger
from sklearn import pipeline

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.model_training.train import (
    dump_cfg_and_metadata,
    get_experiment_id_via_experiment_name,
    log_all_metrics_to_mlflow,
    train_and_validate_model,
)


def train_with_best_model_config(
    cfg: Config,
    logger: Logger,
    metadata: Metadata,
    preprocessor: pipeline.Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> Metadata:
    mlflow.set_experiment(experiment_name=cfg.exp.experiment_name)

    # nested=True because this is nested under a parent train func in main.py.
    run_name: str = "_".join(
        [
            cfg.general.pipeline_name,
            cfg.train.create_model.name,
            str(metadata.processed_num_rows),
            "rows",
            str(cfg.train.num_epochs),
            "epochs",
        ]
    )

    cfg.exp.start_run["run_name"] = run_name
    with mlflow.start_run(**cfg.exp.start_run):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

        metadata = train_and_validate_model(
            cfg=cfg,
            metadata=metadata,
            preprocessor=preprocessor,
            logger=logger,
            X_train=X,
            y_train=y,
            trial=None,  # always None since no more hyperparameter tuning
        )

        mlflow.log_params(metadata.model_artifacts["model_config"])
        mlflow.sklearn.log_model(
            sk_model=metadata.model_artifacts["model"],
            artifact_path="registry",
            signature=metadata.model_artifacts["signature"],
        )

        logger.info("✅ Logged the model to MLflow.")

        overall_performance_val = metadata.model_artifacts["overall_performance_val"]
        logger.info(
            f"✅ Training completed. The model overall_performance is {overall_performance_val}."
        )
        logger.info("✅ Logged the model's overall performance to MLflow.")
        log_all_metrics_to_mlflow(overall_performance_val)

        logger.info("✅ Dumping cfg and metadata to artifacts.")
        dump_cfg_and_metadata(cfg, metadata)

        stores_path = cfg.dirs.stores.base

        mlflow.log_artifacts(
            local_dir=stores_path,
            artifact_path=cfg.exp.log_artifacts["artifact_path"],
        )

        # log to model registry
        experiment_id = get_experiment_id_via_experiment_name(
            experiment_name=cfg.exp.experiment_name
        )

        # FIXME: UNCOMMENT
        # signature = metadata.model_artifacts["signature"]
        # mlflow.models.signature.set_signature(
        #     model_uri=cfg.exp.set_signature["model_uri"].format(
        #         experiment_id=experiment_id, run_id=run_id
        #     ),
        #     signature=signature,
        # )

        model_version = mlflow.register_model(
            model_uri=cfg.exp.register_model["model_uri"].format(
                experiment_id=experiment_id, run_id=run_id
            ),  # this is relative to the run_id! rename to registry to be in sync with local stores
            name=cfg.exp.register_model["name"],
            tags={
                "dev-exp-id": experiment_id,
                "dev-exp-name": cfg.exp.experiment_name,
                "dev-exp-run-id": run_id,
                "dev-exp-run-name": cfg.exp.start_run["run_name"],
            },
        )
        mlflow.log_param("model_version", model_version.version)
        logger.info(
            f"✅ Logged the model to the model registry with version {model_version.version}."
        )

        metadata.set_attrs(
            {
                "run_id": run_id,
                "model_version": model_version.version,
                "experiment_id": experiment_id,
                "run_name": run_name,
                "artifact_uri": mlflow.get_artifact_uri(),
            }
        )

    return metadata
