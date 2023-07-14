"""NOTE: As mentioned, this does not touch on
tricks and tips on how to improve the model performance."""
import copy
import datetime
from typing import Any, Callable, Dict, Tuple, Union

import optuna
from common_utils.core.logger import Logger
from optuna.integration.mlflow import MLflowCallback

from conf.base import Config
from metadata.core import Metadata


def handle_specific_types(obj: Any) -> Any:
    """Handle the specific types for serialization.

    Parameters
    ----------
    obj : Any
        Object to handle.

    Returns
    -------
    Any
        String representation of the object if it's datetime or
        optuna.distributions.BaseDistribution. Otherwise, return the object.
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, optuna.distributions.BaseDistribution):
        return str(obj)
    return obj


def trial_to_dict(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    """Convert a FrozenTrial object to a dictionary.

    Parameters
    ----------
    trial : optuna.trial.FrozenTrial
        The FrozenTrial object.

    Returns
    -------
    Dict[str, Any]
        The FrozenTrial object represented as a dictionary.
    """
    trial_dict: dict = vars(trial).copy()
    for key, value in trial_dict.items():
        if isinstance(value, dict):
            trial_dict[key] = {k: handle_specific_types(v) for k, v in value.items()}
        else:
            trial_dict[key] = handle_specific_types(value)
    return trial_dict


# create an Optuna objective function for hyperparameter tuning
def objective(
    cfg: Config,
    logger: Logger,
    metadata: Metadata,
    partial: Callable,
    trial: optuna.trial._trial.Trial,
) -> float:
    """Objective function for hyperparameter tuning.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    logger : Logger
        Logger for logging.
    metadata : Metadata
        Metadata information.
    partial : Callable
        Partial function.
    trial : optuna.trial._trial.Trial
        Trial object for this iteration.

    Returns
    -------
    float
        Objective function value.
    """
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

    metadata = partial(cfg=cfg, trial=trial, metadata=metadata)

    overall_performance_val = metadata.model_artifacts["overall_performance_val"]
    trial.set_user_attr("val_accuracy", overall_performance_val["val_accuracy"])
    trial.set_user_attr("val_f1", overall_performance_val["val_f1"])
    trial.set_user_attr("val_loss", overall_performance_val["val_loss"])
    return overall_performance_val["val_accuracy"]


def create_pruner(pruner_config: Dict[str, Any]) -> optuna.pruners.BasePruner:
    """Create an Optuna pruner based on the configuration.

    Parameters
    ----------
    pruner_config : Dict[str, Any]
        Configuration for the pruner.

    Returns
    -------
    optuna.pruners.BasePruner
        Optuna pruner object.
    """
    pruner_class = eval(pruner_config.pop("pruner_name"))
    pruner = pruner_class(**pruner_config)
    return pruner


def create_sampler(sampler_config: Dict[str, Any]) -> optuna.samplers.BaseSampler:
    """Create an Optuna sampler based on the configuration.

    Parameters
    ----------
    sampler_config : Dict[str, Any]
        Configuration for the sampler.

    Returns
    -------
    optuna.samplers.BaseSampler
        Optuna sampler object.
    """
    sampler_class = eval(sampler_config.pop("sampler_name"))
    sampler = sampler_class(**sampler_config)
    return sampler


def create_optuna_study(cfg: Config, logger: Logger) -> optuna.study.Study:
    """Create an Optuna study based on the configuration.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    logger : Logger
        Logger for logging.

    Returns
    -------
    optuna.study.Study
        Optuna study object.
    """
    pruner = create_pruner(cfg.train.optimize.pruner)
    sampler = create_sampler(cfg.train.optimize.sampler)

    logger.info("Creating an Optuna study.")
    logger.warning(
        "Remember to check `create_study.direction` to ensure"
        "you set `maximize` or `minimize` correctly."
    )

    study = optuna.create_study(
        pruner=pruner, sampler=sampler, **cfg.train.optimize.create_study
    )
    return study


def run_optimization(
    study: optuna.study.Study,
    cfg: Config,
    logger: Logger,
    metadata: Metadata,
    partial: Callable,
) -> optuna.study.Study:
    """Run optimization on the study.

    Parameters
    ----------
    study : optuna.study.Study
        Optuna study object.
    cfg : Config
        Configuration object.
    logger : Logger
        Logger for logging.
    metadata : Metadata
        Metadata information.
    partial : Callable
        Partial function.

    Returns
    -------
    optuna.study.Study
        Optuna study object after optimization.
    """
    mlflow_callback = MLflowCallback(
        tracking_uri=cfg.exp.tracking_uri, metric_name="val_loss"  # FIXME: hardcoded
    )
    study.optimize(
        lambda trial: objective(
            cfg=cfg,
            logger=logger,
            metadata=metadata,
            partial=partial,
            trial=trial,
        ),
        n_trials=cfg.train.optimize.n_trials,
        callbacks=[mlflow_callback],
    )
    return study


def process_optimization_results(
    study: optuna.study.Study,
    metadata: Metadata,
    cfg: Config,
) -> Tuple[Metadata, Config]:
    """Process the optimization results.

    Parameters
    ----------
    study : optuna.study.Study
        Optuna study object.
    metadata : Metadata
        Metadata information.
    cfg : Config
        Configuration object.

    Returns
    -------
    Tuple[Metadata, Config]
        Metadata and configuration after processing the results.
    """
    trials = study.trials
    trials_dict = {trial.number: trial_to_dict(trial) for trial in trials}

    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(by=["user_attrs_val_loss"], ascending=False)

    best_params = study.best_trial.params
    best_params["best_trial"] = study.best_trial.number

    metadata.set_attrs(
        {
            "best_params": best_params,
            "trials": trials_dict,
            "trials_df": trials_df,
        }
    )
    cfg.train.create_model.alpha = study.best_params["model__alpha"]
    cfg.train.create_model.power_t = study.best_params["model__power_t"]
    return metadata, cfg


def optimize(
    cfg: Config,
    logger: Logger,
    metadata: Metadata,
    partial: Callable,
) -> Union[Metadata, Config]:
    """Perform the optimization process.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    logger : Logger
        Logger for logging.
    metadata : Metadata
        Metadata information.
    partial : Callable
        Partial function.

    Returns
    -------
    Union[Metadata, Config]
        Metadata and configuration after optimization.
    """
    logger.info(
        """Seeing inside objective function as well to ensure the hyperparam grid is seeded.
        See https://optuna.readthedocs.io/en/stable/faq.html for how to seed in Optuna"""
    )
    study = create_optuna_study(cfg, logger)
    study = run_optimization(study, cfg, logger, metadata, partial)
    metadata, cfg = process_optimization_results(study, metadata, cfg)
    return metadata, cfg
