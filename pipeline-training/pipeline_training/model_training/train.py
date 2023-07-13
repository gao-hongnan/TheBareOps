import pickle
import warnings
from importlib import import_module
from typing import Any, Dict, Optional, Tuple, Union

import mlflow
import numpy as np
import optuna
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger
from sklearn import pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from conf.base import Config
from conf.train.base import CreateBaselineModel, CreateModel
from metadata.core import Metadata
from pipeline_training.model_training.preprocess import Preprocessor

warnings.filterwarnings("ignore")


def create_model_from_config(
    model_config: Union[CreateModel, CreateBaselineModel]
) -> BaseEstimator:
    model_args = model_config.model_dump(mode="python")
    model_name = model_args.pop("name")
    module_name, class_name = model_name.rsplit(".", 1)
    module = import_module(module_name)
    model_class = getattr(module, class_name)
    model = model_class(**model_args)
    return model


def calculate_classification_metrics(
    y: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prefix: str = "val",
) -> Dict[str, float]:
    """Predict on holdout set."""
    # TODO: make metrics an abstract object instead of dict
    performance = {"overall": {}, "report": {}, "per_class": {}}

    classes = np.unique(y)
    num_classes = len(classes)

    prf_metrics = precision_recall_fscore_support(y, y_pred, average="weighted")
    test_loss = log_loss(y, y_prob)
    test_accuracy = accuracy_score(y, y_pred)
    test_balanced_accuracy = balanced_accuracy_score(y, y_pred)

    # Brier score
    if num_classes == 2:
        test_brier_score = brier_score_loss(y, y_prob[:, 1])
        test_roc_auc = roc_auc_score(y, y_prob[:, 1])
    else:
        test_brier_score = np.mean(
            [brier_score_loss(y == i, y_prob[:, i]) for i in range(num_classes)]
        )
        test_roc_auc = roc_auc_score(y, y_prob, multi_class="ovr")

    overall_performance = {
        f"{prefix}_loss": test_loss,
        f"{prefix}_precision": prf_metrics[0],
        f"{prefix}_recall": prf_metrics[1],
        f"{prefix}_f1": prf_metrics[2],
        f"{prefix}_accuracy": test_accuracy,
        f"{prefix}_balanced_accuracy": test_balanced_accuracy,
        f"{prefix}_roc_auc": test_roc_auc,
        f"{prefix}_brier_score": test_brier_score,
    }
    performance["overall"] = overall_performance

    test_confusion_matrix = confusion_matrix(y, y_pred)
    test_classification_report = classification_report(
        y, y_pred, output_dict=True
    )  # output_dict=True to get result as dictionary

    performance["report"] = {
        "test_confusion_matrix": test_confusion_matrix,
        "test_classification_report": test_classification_report,
    }

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(
        y, y_pred, average=None
    )  # None to get per-class metrics

    for i, _class in enumerate(classes):
        performance["per_class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }
    return performance


def log_all_metrics_to_mlflow(metrics: Dict[str, Any]) -> None:
    """Log all metrics to MLFlow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def dump_cfg_and_metadata(cfg: Config, metadata: Metadata) -> None:
    """Dump cfg and metadata to artifacts."""
    with open(f"{cfg.dirs.stores.artifacts}/cfg.pkl", "wb") as file:
        pickle.dump(cfg, file)

    with open(f"{cfg.dirs.stores.artifacts}/metadata.pkl", "wb") as file:
        pickle.dump(metadata, file)


def get_experiment_id_via_experiment_name(experiment_name: str) -> int:
    """Get experiment ID via experiment name."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    return experiment_id


# NOTE: the training loop here is moot since there is `max_iters`, but just
# want to test the code in MLFlow.
class Trainer:
    def __init__(
        self,
        cfg: Config,
        logger: Logger,
        metadata: Metadata,
        preprocessor: Preprocessor,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.metadata = metadata
        self.preprocessor = preprocessor

    def create_model_from_config(
        self, model_config: Union[CreateModel, CreateBaselineModel]
    ) -> BaseEstimator:
        model_args = model_config.model_dump(mode="python")
        model_name = model_args.pop("name")
        module_name, class_name = model_name.rsplit(".", 1)
        module = import_module(module_name)
        model_class = getattr(module, class_name)
        model = model_class(**model_args)
        return model

    def create_baseline_model(self) -> BaseEstimator:
        return self.create_model_from_config(self.cfg.train.create_baseline_model)

    def create_model(self) -> BaseEstimator:
        return self.create_model_from_config(self.cfg.train.create_model)

    def train_model(
        self,
        trial: Optional[optuna.trial._trial.Trial] = None,
    ) -> Dict[str, Any]:
        """Train model."""
        seed = seed_all(self.cfg.general.seed, seed_torch=False)
        self.logger.info(f"Using seed {seed}")

        self.logger.info("Training model...")
        X_train, y_train = self.metadata.X_train, self.metadata.y_train
        X_val, y_val = self.metadata.X_val, self.metadata.y_val

        model = self.create_model()

        # Training
        for epoch in range(self.cfg.train.num_epochs):
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_prob_train = model.predict_proba(X_train)

            y_pred_val = model.predict(X_val)
            y_prob_val = model.predict_proba(X_val)

            performance_train = calculate_classification_metrics(
                y=y_train,
                y_pred=y_pred_train,
                y_prob=y_prob_train,
                prefix="train",
            )
            performance_val = calculate_classification_metrics(
                y=y_val, y_pred=y_pred_val, y_prob=y_prob_val, prefix="val"
            )

            # Log performance metrics for the current epoch
            if not trial:  # if not hyperparameter tuning then we log to mlflow
                mlflow.log_metrics(
                    metrics={
                        **performance_train["overall"],
                        **performance_val["overall"],
                    },
                    step=epoch,
                )

            if not epoch % self.cfg.train.log_every_n_epoch:
                self.logger.info(
                    f"Epoch: {epoch:02d} | "
                    f"train_loss: {performance_train['overall']['train_loss']:.5f}, "
                    f"val_loss: {performance_val['overall']['val_loss']:.5f}, "
                    f"val_accuracy: {performance_val['overall']['val_accuracy']:.5f}"
                )

        # Log the model with a signature that defines the schema of the model's inputs and outputs.
        # When the model is deployed, this signature will be used to validate inputs.
        if not trial:
            self.logger.info(
                "This is not in a trial, it is likely training a final model with the best hyperparameters"
            )
            signature = mlflow.models.infer_signature(X_val, model.predict(X_val))

        model_artifacts = {
            "preprocessor": self.preprocessor,
            "model": model,
            "overall_performance_train": performance_train["overall"],
            "report_performance_train": performance_train["report"],
            "per_class_performance_train": performance_train["per_class"],
            "overall_performance_val": performance_val["overall"],
            "report_performance_val": performance_val["report"],
            "per_class_performance_val": performance_val["per_class"],
            "signature": signature if not trial else None,
            "model_config": model.get_params(),
        }
        self.metadata.set_attrs({"model_artifacts": model_artifacts})
        return self.metadata

    def train(self):
        mlflow.set_experiment(experiment_name=self.cfg.exp.experiment_name)

        # nested=True because this is nested under a parent train func in main.py.
        with mlflow.start_run(**self.cfg.exp.start_run):
            run_id = mlflow.active_run().info.run_id
            self.logger.info(f"MLflow run_id: {run_id}")

            metadata = self.train_model(trial=None)
            mlflow.sklearn.log_model(
                sk_model=metadata.model_artifacts["model"],
                artifact_path="registry",
                signature=metadata.model_artifacts["signature"],
            )

            self.logger.info("✅ Logged the model to MLflow.")

            overall_performance_val = metadata.model_artifacts[
                "overall_performance_val"
            ]
            self.logger.info(
                f"✅ Training completed. The model overall_performance is {overall_performance_val}."
            )
            self.logger.info("✅ Logged the model's overall performance to MLflow.")
            log_all_metrics_to_mlflow(overall_performance_val)

            self.logger.info("✅ Dumping cfg and metadata to artifacts.")
            dump_cfg_and_metadata(self.cfg, metadata)

            stores_path = self.cfg.dirs.stores.base

            mlflow.log_artifacts(
                local_dir=stores_path,
                artifact_path=self.cfg.exp.log_artifacts["artifact_path"],
            )

            # log to model registry
            # log to model registry
            experiment_id = get_experiment_id_via_experiment_name(
                experiment_name=self.cfg.exp.experiment_name
            )

            # FIXME: UNCOMMENT
            # signature = metadata.model_artifacts["signature"]
            # mlflow.models.signature.set_signature(
            #     model_uri=self.cfg.exp.set_signature["model_uri"].format(
            #         experiment_id=experiment_id, run_id=run_id
            #     ),
            #     signature=signature,
            # )

            model_version = mlflow.register_model(
                model_uri=self.cfg.exp.register_model["model_uri"].format(
                    experiment_id=experiment_id, run_id=run_id
                ),  # this is relative to the run_id! rename to registry to be in sync with local stores
                name=self.cfg.exp.register_model["name"],
                tags={
                    "dev-exp-id": experiment_id,
                    "dev-exp-name": self.cfg.exp.experiment_name,
                    "dev-exp-run-id": run_id,
                    "dev-exp-run-name": self.cfg.exp.start_run["run_name"],
                },
            )
            mlflow.log_param("model_version", model_version.version)
            self.logger.info(
                f"✅ Logged the model to the model registry with version {model_version.version}."
            )

        return metadata
