def train_and_validate_model(
    cfg: Config,
    model,
    logger: Logger,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    trial: Optional[optuna.trial._trial.Trial] = None,
) -> Dict[str, Any]:
    """Train model."""
    seed = seed_all(cfg.general.seed, seed_torch=False)
    logger.info(f"Using seed {seed}")

    logger.info("Training model...")
    X_train, y_train = metadata.X_train, metadata.y_train
    # X_val, y_val = metadata.X_val, metadata.y_val

    # Training
    for epoch in range(cfg.train.num_epochs):
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

        if not epoch % cfg.train.log_every_n_epoch:
            logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {performance_train['overall']['train_loss']:.5f}, "
                f"val_loss: {performance_val['overall']['val_loss']:.5f}, "
                f"val_accuracy: {performance_val['overall']['val_accuracy']:.5f}"
            )

    # Log the model with a signature that defines the schema of the model's inputs and outputs.
    # When the model is deployed, this signature will be used to validate inputs.
    if not trial:
        logger.info(
            "This is not in a trial, it is likely training a final model with the best hyperparameters"
        )
        signature = mlflow.models.infer_signature(X_val, model.predict(X_val))

    model_artifacts = {
        "preprocessor": preprocessor,
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
    metadata.set_attrs({"model_artifacts": model_artifacts})
    return metadata
