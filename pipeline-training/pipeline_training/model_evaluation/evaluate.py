"""For more info see my notebooks on model evaluation."""

from typing import Optional

import mlflow
import numpy as np
from common_utils.core.logger import Logger
from sklearn.base import BaseEstimator

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.model_training.train import calculate_classification_metrics


# TODO: dump confusion matrix and classification report to image/media.
def predict_on_holdout_set(
    cfg: Config,
    metadata: Metadata,
    logger: Logger,
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_id: Optional[str] = None,
) -> Metadata:
    """Predict on holdout set."""
    logger.info("Predicting on holdout set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    performance = calculate_classification_metrics(
        y=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        **cfg.evaluate.predict_on_holdout_set.model_dump(mode="python"),
    )

    if run_id is not None:
        # Log metrics to the same MLflow session
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics=performance["overall"], step=None)

    metadata.set_attrs(attr_dict={"holdout_performance": performance})
    return metadata


# Sebastian Raschka 2014-2023
# mlxtend Machine Learning Library Extensions
#
# Nonparametric Permutation Test
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(
        sample_indices, size=sample_indices.shape[0], replace=True
    )
    return X[bootstrap_indices], y[bootstrap_indices]


def bias_variance_decomp(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    loss="0-1_loss",
    num_rounds=200,
    random_seed=None,
    **fit_params,
):
    """
    estimator : object
        A classifier or regressor object or class implementing both a
        `fit` and `predict` method similar to the scikit-learn API.

    X_train : array-like, shape=(num_examples, num_features)
        A training dataset for drawing the bootstrap samples to carry
        out the bias-variance decomposition.

    y_train : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_train` examples.

    X_test : array-like, shape=(num_examples, num_features)
        The test dataset for computing the average loss, bias,
        and variance.

    y_test : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_test` examples.

    loss : str (default='0-1_loss')
        Loss function for performing the bias-variance decomposition.
        Currently allowed values are '0-1_loss' and 'mse'.

    num_rounds : int (default=200)
        Number of bootstrap rounds (sampling from the training set)
        for performing the bias-variance decomposition. Each bootstrap
        sample has the same size as the original training set.

    random_seed : int (default=None)
        Random seed for the bootstrap sampling used for the
        bias-variance decomposition.

    fit_params : additional parameters
        Additional parameters to be passed to the .fit() function of the
        estimator when it is fit to the bootstrap samples.

    Returns
    ----------
    avg_expected_loss, avg_bias, avg_var : returns the average expected
        average bias, and average bias (all floats), where the average
        is computed over the data points in the test set.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/

    """
    supported = ["0-1_loss", "mse"]
    if loss not in supported:
        raise NotImplementedError("loss must be one of the following: %s" % supported)

    for ary in (X_train, y_train, X_test, y_test):
        if hasattr(ary, "loc"):
            raise ValueError(
                "The bias_variance_decomp does not "
                "support pandas DataFrames yet. "
                "Please check the inputs to "
                "X_train, y_train, X_test, y_test. "
                "If e.g., X_train is a pandas "
                "DataFrame, try passing it as NumPy array via "
                "X_train=X_train.values."
            )

    rng = np.random.RandomState(random_seed)  # pylint: disable=no-member

    if loss == "0-1_loss":
        dtype = np.int64
    elif loss == "mse":
        dtype = np.float64

    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=dtype)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)

        # Keras support
        if estimator.__class__.__name__ in ["Sequential", "Functional"]:
            # reset model
            for ix, layer in enumerate(estimator.layers):
                if hasattr(estimator.layers[ix], "kernel_initializer") and hasattr(
                    estimator.layers[ix], "bias_initializer"
                ):
                    weight_initializer = estimator.layers[ix].kernel_initializer
                    bias_initializer = estimator.layers[ix].bias_initializer

                    old_weights, old_biases = estimator.layers[ix].get_weights()

                    estimator.layers[ix].set_weights(
                        [
                            weight_initializer(shape=old_weights.shape),
                            bias_initializer(shape=len(old_biases)),
                        ]
                    )

            estimator.fit(X_boot, y_boot, **fit_params)
            pred = estimator.predict(X_test).reshape(1, -1)
        else:
            pred = estimator.fit(X_boot, y_boot, **fit_params).predict(X_test)
        all_pred[i] = pred

    if loss == "0-1_loss":
        main_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=all_pred
        )

        avg_expected_loss = np.apply_along_axis(
            lambda x: (x != y_test).mean(), axis=1, arr=all_pred
        ).mean()

        avg_bias = np.sum(main_predictions != y_test) / y_test.size

        var = np.zeros(pred.shape)

        for pred in all_pred:
            # NOTE: change from np.int to int64 due to np version issues
            var += (pred != main_predictions).astype(np.int64)
        var /= num_rounds
        avg_var = var.sum() / y_test.shape[0]
    else:
        avg_expected_loss = np.apply_along_axis(
            lambda x: ((x - y_test) ** 2).mean(), axis=1, arr=all_pred
        ).mean()

        main_predictions = np.mean(all_pred, axis=0)

        avg_bias = np.sum((main_predictions - y_test) ** 2) / y_test.size
        avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var


def bias_variance(
    cfg: Config,
    metadata: Metadata,
    logger: Logger,
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_id: Optional[str] = None,
) -> Metadata:
    """We use the `mlxtend` library to estimate the Bias-Variance Tradeoff in
    our final model. The core idea behind this function is to use bagging
    and repeatedly sample from our training set so as to simulate that we are
    actually drawing samples from the "true" population over a distribution $\mathcal{P}$.
    """

    avg_expected_loss, avg_bias, avg_variance = bias_variance_decomp(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        **cfg.evaluate.bias_variance.model_dump(mode="python"),
    )

    if run_id is not None:
        # Log metrics to the same MLflow session
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(
                metrics={
                    "avg_expected_loss": avg_expected_loss,
                    "avg_bias": avg_bias,
                    "avg_variance": avg_variance,
                },
                step=None,
            )

    logger.info(f"Average expected loss: {avg_expected_loss}")
    logger.info(f"Average bias: {avg_bias}")
    logger.info(f"Average variance: {avg_variance}")

    metadata.set_attrs(
        {
            "avg_expected_loss": avg_expected_loss,
            "avg_bias": avg_bias,
            "avg_variance": avg_variance,
        }
    )
    return metadata


def benefit_structure():
    """Calculate benefit structure."""


def find_auc_threshold():
    """Find the threshold that maximizes the AUC."""
