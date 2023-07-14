"""See https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning.

Here I added a bunch of dummy/empty functions fo reference and the sake
of completeness.

Note the evaluate_model_performance is already implemented in
the evaluate.py file in the pipeline_training/model_evaluation folder.

And compare_with_currently_deployed_model will be inside promote.py.

Do this step before promoting the model to production.
"""


def evaluate_model_performance(model, X_test, y_test):
    """
    Evaluate the performance of the trained model on the test dataset.

    Parameters
    ----------
    model : object
        Trained model object.
    X_test : array-like
        Testing feature data.
    y_test : array-like
        Testing target data.

    Returns
    -------
    dict
        Dictionary of computed evaluation metrics.
    """
    pass


def compare_with_baseline(model_metrics, baseline_metrics):
    """
    Compare the performance metrics of the new model with the baseline model.

    Parameters
    ----------
    model_metrics : dict
        Computed evaluation metrics for the new model.
    baseline_metrics : dict
        Computed evaluation metrics for the baseline model.

    Returns
    -------
    dict
        Dictionary comparing new model and baseline model.
    """
    pass


def compare_with_currently_deployed_model(model_metrics, deployed_model_metrics):
    """
    Compare performance metrics of the new model with the currently deployed model.

    Parameters
    ----------
    model_metrics : dict
        Computed evaluation metrics for the new model.
    deployed_model_metrics : dict
        Computed evaluation metrics for the currently deployed model.

    Returns
    -------
    dict
        Dictionary comparing new model and currently deployed model.
    """
    pass


def test_model_consistency(model, data_segments):
    """
    Check performance consistency of the model across various data segments.

    Parameters
    ----------
    model : object
        Trained model object.
    data_segments : list
        List of data segments to test the model on.

    Returns
    -------
    dict
        Dictionary with model performance on each data segment.
    """
    pass


def test_model_deployment_compatibility(model):
    """
    Test if the model is compatible with the deployment infrastructure.

    Parameters
    ----------
    model : object
        Trained model object.

    Returns
    -------
    bool
        True if the model is compatible with the infrastructure, False otherwise.
    """
    pass


def test_prediction_api_compatibility(model):
    """
    Test if the model's outputs are compatible with the prediction service API.

    Parameters
    ----------
    model : object
        Trained model object.

    Returns
    -------
    bool
        True if the model's outputs are compatible with the API, False otherwise.
    """
    pass


def offline_cross_validation(model, X, y, cv):
    """
    Perform cross validation on the model, used in offline model validation.

    Parameters
    ----------
    model : object
        Model object.
    X : array-like
        Feature data.
    y : array-like
        Target data.
    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.

    Returns
    -------
    list
        List of scores for each fold.
    """
    pass


def offline_roc_auc_score(model, X, y):
    """
    Calculate the ROC AUC score for classification models in offline validation.

    Parameters
    ----------
    model : object
        Model object.
    X : array-like
        Feature data.
    y : array-like
        Target data.

    Returns
    -------
    float
        ROC AUC score of the model.
    """
    pass


def online_predictive_performance(model, X_realtime, y_realtime):
    """
    Evaluate the predictive performance of the model on real-time data.

    Parameters
    ----------
    model : object
        Model object.
    X_realtime : array-like
        Real-time feature data.
    y_realtime : array-like
        Real-time target data.

    Returns
    -------
    dict
        Dictionary of computed evaluation metrics.
    """
    pass


def online_ab_testing(model_A, model_B, X_realtime, y_realtime):
    """
    Conduct A/B testing between two models using real-time data.

    Parameters
    ----------
    model_A : object
        First model object for testing.
    model_B : object
        Second model object for testing.
    X_realtime : array-like
        Real-time feature data.
    y_realtime : array-like
        Real-time target data.

    Returns
    -------
    dict
        Dictionary containing comparison metrics between model_A and model_B.
    """
    pass


def online_model_monitoring(model, X_realtime, y_realtime):
    """
    Monitor the performance of the model over time with real-time data.

    Parameters
    ----------
    model : object
        Model object.
    X_realtime : array-like
        Real-time feature data.
    y_realtime : array-like
        Real-time target data.

    Returns
    -------
    dict or DataFrame
        Model performance metrics recorded over time.
    """
    pass
