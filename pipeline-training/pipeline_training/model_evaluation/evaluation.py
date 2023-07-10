# Evaluate on test data
# performance = predict_on_holdout_set(model, X_test, y_test)

# TODO: dump confusion matrix and classification report to image/media.
def predict_on_holdout_set(
    model, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    """Predict on holdout set."""
    # TODO: make metrics an abstract object instead of dict
    performance = {"overall": {}, "report": {}, "per_class": {}}

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    classes = np.unique(y_test)
    num_classes = len(classes)

    prf_metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    test_loss = log_loss(y_test, y_prob)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # Brier score
    if num_classes == 2:
        test_brier_score = brier_score_loss(y_test, y_prob[:, 1])
        test_roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        test_brier_score = np.mean(
            [brier_score_loss(y_test == i, y_prob[:, i]) for i in range(num_classes)]
        )
        test_roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    overall_performance = {
        "test_loss": test_loss,
        "test_precision": prf_metrics[0],
        "test_recall": prf_metrics[1],
        "test_f1": prf_metrics[2],
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced_accuracy,
        "test_roc_auc": test_roc_auc,
        "test_brier_score": test_brier_score,
    }
    performance["overall"] = overall_performance

    test_confusion_matrix = confusion_matrix(y_test, y_pred)
    test_classification_report = classification_report(
        y_test, y_pred, output_dict=True
    )  # output_dict=True to get result as dictionary

    performance["report"] = {
        "test_confusion_matrix": test_confusion_matrix,
        "test_classification_report": test_classification_report,
    }

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )  # None to get per-class metrics

    for i, _class in enumerate(classes):
        performance["per_class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }
    return performance
