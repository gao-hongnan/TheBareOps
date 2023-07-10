"""NOTE: As mentioned, this does not touch on
tricks and tips on how to improve the model performance."""
import copy
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Union

import optuna
from optuna.integration.mlflow import MLflowCallback
from rich.pretty import pprint

from metadata.core import Metadata
from pipeline_training.model_training.train import Trainer


# create an Optuna objective function for hyperparameter tuning
def objective(
    cfg,
    logger,
    metadata,
    preprocessor,
    trial: optuna.trial._trial.Trial,
):
    # define hyperparameters to tune
    # the actual hyperparameters depend on your model
    logger.warning("Performing a deepcopy of the config object to avoid mutation.")
    cfg = copy.deepcopy(cfg)

    cfg.train.create_model.alpha = trial.suggest_loguniform(
        "model__alpha",
        *cfg.train.optimize.hyperparams_grid["model__alpha"],
    )
    cfg.train.create_model.power_t = trial.suggest_uniform(
        "model__power_t", *cfg.train.optimize.hyperparams_grid["model__power_t"]
    )

    # you assign back to cfg.train.model and vectorizer so train_model can use them
    # train and evaluate the model using the current hyperparameters
    trainer = Trainer(
        cfg=cfg, logger=logger, metadata=metadata, preprocessor=preprocessor
    )

    metadata = trainer.train_model(trial=trial)

    overall_performance_val = metadata.model_artifacts["overall_performance_val"]
    trial.set_user_attr("val_accuracy", overall_performance_val["val_accuracy"])
    trial.set_user_attr("val_f1", overall_performance_val["val_f1"])
    trial.set_user_attr("val_loss", overall_performance_val["val_loss"])
    return overall_performance_val["val_accuracy"]


def create_pruner(pruner_config: Dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_class = eval(pruner_config.pop("pruner_name"))
    pruner = pruner_class(**pruner_config)
    return pruner


def create_sampler(sampler_config: Dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_class = eval(sampler_config.pop("sampler_name"))
    sampler = sampler_class(**sampler_config)
    return sampler


def optimize(cfg, logger, metadata, preprocessor) -> Union[Metadata, SimpleNamespace]:
    logger.info(
        """Seeing inside objective function as well to ensure the hyperparam grid is seeded.
        See https://optuna.readthedocs.io/en/stable/faq.html for how to seed in Optuna"""
    )

    pruner = create_pruner(cfg.train.optimize.pruner)
    sampler = create_sampler(cfg.train.optimize.sampler)
    study = optuna.create_study(
        pruner=pruner, sampler=sampler, **cfg.train.optimize.create_study
    )
    mlflow_callback = MLflowCallback(
        tracking_uri=cfg.exp.tracking_uri, metric_name="val_loss"
    )

    study.optimize(
        lambda trial: objective(
            cfg=cfg,
            logger=logger,
            metadata=metadata,
            trial=trial,
            preprocessor=preprocessor,
        ),
        n_trials=cfg.train.optimize.n_trials,
        callbacks=[mlflow_callback],
    )

    # print the best hyperparameters
    trials_df = study.trials_dataframe()
    pprint(trials_df)
    trials_df = trials_df.sort_values(by=["user_attrs_val_loss"], ascending=False)

    metadata.best_params = {**study.best_trial.params}
    metadata.best_params["best_trial"] = study.best_trial.number

    # update the config object with the best hyperparameters
    # FIXME: here is hardcoded and is prone to error

    cfg.train.create_model.alpha = study.best_params["model__alpha"]
    cfg.train.create_model.power_t = study.best_params["model__power_t"]
    return metadata, cfg
