import pickle
import warnings
from importlib import import_module
from typing import Any, Dict, Optional, Tuple, Union
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
from pipeline_training.model_training.preprocess import Preprocessor
import mlflow
import numpy as np
import optuna
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger
from sklearn.base import BaseEstimator
from sklearn import pipeline
from conf.train.base import CreateModel, CreateBaselineModel
from conf.base import Config
from metadata.core import Metadata

warnings.filterwarnings("ignore")


def create_baseline_model(cfg: Config) -> BaseEstimator:
    model_args = cfg.train.create_baseline_model.model_dump(mode="python")
    model_name = model_args.pop("name")
    module_name, class_name = model_name.rsplit(".", 1)
    module = import_module(module_name)
    model_class = getattr(module, class_name)
    model = model_class(**model_args)
    return model


def create_model(cfg: Config) -> BaseEstimator:
    model_args = cfg.train.create_model.model_dump(mode="python")
    model_name = model_args.pop("name")
    module_name, class_name = model_name.rsplit(".", 1)
    module = import_module(module_name)
    model_class = getattr(module, class_name)
    model = model_class(**model_args)
    return model


def train_model(
    cfg: Config,
    logger: Logger,
    metadata: Metadata,
    model: BaseEstimator,
    # preprocessor: pipeline.Pipeline,
    trial: Optional[optuna.trial._trial.Trial] = None,
) -> Dict[str, Any]:
    """Train model."""
    seed_all(cfg.general.seed, seed_torch=False)

    logger.info("Training model...")
    X_train, y_train = metadata.X_train, metadata.y_train
    X_val, y_val = metadata.X_val, metadata.y_val

    # Training
    for epoch in range(cfg.train.num_epochs):
        model.fit(X_train, y_train)

        train_loss = log_loss(y_train, model.predict_proba(X_train))
        # evaluate on validation data
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        val_accuracy = accuracy_score(y_val, model.predict(X_val))

        # Log performance metrics for the current epoch
        if not trial:  # if not hyperparameter tuning then we log to mlflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )

        if not epoch % cfg.train.log_every_n_epoch:
            logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}, "
                f"val_accuracy: {val_accuracy:.5f}"
            )

    # Evaluate on test data
    # performance = predict_on_holdout_set(model, X_test, y_test)

    # Log the model with a signature that defines the schema of the model's inputs and outputs.
    # When the model is deployed, this signature will be used to validate inputs.
    # if not trial:
    #     logger.info(
    #         "This is not in a trial, it is likely training a final model with the best hyperparameters"
    #     )
    #     signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

    model_artifacts = {
        # "vectorizer": vectorizer,
        "model": model,
        # "overall_performance": performance["overall"],
        # "report_performance": performance["report"],
        # "per_class_performance": performance["per_class"],
        # "signature": signature if not trial else None,
        "val_loss": val_loss,
        "model_config": model.get_params(),
    }
    metadata.model_artifacts = model_artifacts
    return metadata


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
        self, trial: Optional[optuna.trial._trial.Trial] = None
    ) -> Dict[str, Any]:
        """Train model."""
        seed_all(self.cfg.general.seed, seed_torch=False)

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
                    f"val_loss: {performance_val['overall']['train_loss']:.5f}, "
                    f"val_accuracy: {performance_val['overall']['accuracy']:.5f}"
                )

        # Evaluate on test data
        # performance = predict_on_holdout_set(model, X_test, y_test)

        # Log the model with a signature that defines the schema of the model's inputs and outputs.
        # When the model is deployed, this signature will be used to validate inputs.
        # if not trial:
        #     logger.info(
        #         "This is not in a trial, it is likely training a final model with the best hyperparameters"
        #     )
        #     signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

        model_artifacts = {
            "preprocessor": self.preprocessor,
            "model": model,
            "overall_performance_train": performance_train["overall"],
            "report_performance_train": performance_train["report"],
            "per_class_performance_train": performance_train["per_class"],
            "overall_performance_val": performance_val["overall"],
            "report_performance_val": performance_val["report"],
            "per_class_performance_val": performance_val["per_class"],
            # "signature": signature if not trial else None,
            "model_config": model.get_params(),
        }
        self.metadata.set_attrs(model_artifacts)
        return self.metadata
