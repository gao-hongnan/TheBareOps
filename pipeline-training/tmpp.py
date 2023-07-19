import mlflow
import time

client = mlflow.tracking.MlflowClient(
    tracking_uri="http://mlflow:mlflow@34.142.130.3:5005/"
)
# exp_id = client.get_experiment_by_name("thebareops_sgd_study").experiment_id
# client.delete_experiment(exp_id)
# client.delete_experiment(exp_id)
# client.delete_experiment("3")
client.transition_model_version_stage(
    name="thebareops_sgd",
    version="3",
    stage="None",  # Or "Staging", "Archived"
)

time.sleep(100)


cfg = Config(
    exp=Experiment(
        tracking_uri="http://mlflow:mlflow@34.142.130.3:5005/",
        set_signature={
            "model_uri": "gs://engr-nscc-mlflow-bucket/artifacts/{experiment_id}/{run_id}/artifacts/registry"
        },
    )
)

logger = Logger(
    log_file="pipeline_training.log",
    log_root_dir=cfg.dirs.stores.logs,
    module_name=__name__,
    propagate=False,
    level=logging.DEBUG,
).logger


metadata = Metadata()

connection = BigQuery(
    project_id=cfg.env.project_id,
    google_application_credentials=cfg.env.google_application_credentials,
    dataset=cfg.env.bigquery_raw_dataset,
    table_name=cfg.env.bigquery_raw_table_name,
)

# NOTE: extract.py
extract = Extract(cfg=cfg, metadata=metadata, logger=logger, connection=connection)
metadata = extract.run()


# NOTE: ??? validate_raw, so if here fails, maybe we should not move on to load.
expected_raw_schema = RawSchema.to_pd_dtypes()

# NOTE: at this stage the schema reflected that the data types were not correct
# for example, our number_of_trades is Int64, but it should be int64.
validator = DataFrameValidator(df=metadata.raw_df, schema=expected_raw_schema)
validator.validate_schema().validate_data_types().validate_missing()

# NOTE: start of load.py
storage = GCS(
    project_id=cfg.env.project_id,
    google_application_credentials=cfg.env.google_application_credentials,
    bucket_name=cfg.env.gcs_bucket_name,
)
raw_dvc = SimpleDVC(
    storage=storage,
    remote_bucket_project_name=cfg.env.gcs_bucket_project_name,
    data_dir=cfg.dirs.data.raw,
    metadata_dir=cfg.dirs.stores.blob.raw,
)

load = Load(cfg=cfg, metadata=metadata, logger=logger, dvc=raw_dvc)
metadata = load.run()


# NOTE: clean.py
clean_dvc = SimpleDVC(
    storage=storage,
    remote_bucket_project_name=cfg.env.gcs_bucket_project_name,
    data_dir=cfg.dirs.data.processed,
    metadata_dir=cfg.dirs.stores.blob.processed,
)
clean = Clean(cfg=cfg, metadata=metadata, logger=logger, dvc=clean_dvc)
metadata = clean.run()


# NOTE: ??? validate_raw, so if here fails, maybe we should not move on to load.
expected_cleaned_schema = CleanedSchema.to_pd_dtypes()

# NOTE: Checking processed df is sufficient because X and y are derived from
# it, so if it is correct, then X and y are correct.
validator = DataFrameValidator(df=metadata.processed_df, schema=expected_cleaned_schema)
validator.validate_schema().validate_data_types().validate_missing()

# NOTE: resampling.py
# TODO: Consider remove X and y from init of Resampler since it can be obtained
# from metadata.
X, y = metadata.X, metadata.y
resampler = Resampler(cfg=cfg, metadata=metadata, logger=logger, X=X, y=y)

# NOTE: Subsetting resampler works why? Because of __getitem__ method.
X_train, y_train = resampler["train"]
X_val, y_val = resampler["val"]
X_test, y_test = resampler["test"]

# NOTE: resampler.metadata is updated after resampling.
metadata = resampler.metadata

# TODO: Inject resample to train or pipeline?

# Assume splits is a dictionary of your data splits
splits = {
    "train": X_train,
    "val": X_val,
    "test": X_test,
}

total_size = len(X_train) + len(X_val) + len(X_test)
table = log_data_splits_summary(splits, total_size)
logger.info(f"Data splits summary:\n{table}")


# NOTE: model_training/preprocess.py
preprocessor = Preprocessor(
    cfg=cfg, metadata=metadata, logger=logger
).create_preprocessor()
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)
y_test = label_encoder.transform(y_test)

# NOTE: overwriting X_train, X_val, X_test, y_train, y_val, y_test
metadata.set_attrs(
    {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "classes": label_encoder.classes_,
        "num_classes": len(label_encoder.classes_),
        "label_encoder": label_encoder,
    }
)


# TODO: Technically, BASELINE MODEL can be rule based or something simple for
# ml models to beat.
# Check https://github.com/gao-hongnan/aiap-batch10-coronary-artery-disease/tree/master/notebooks
# for more details.
# So the idea is before the pipeline run in production

# In reality, this step can select like a list of [model1, model2, model3]
# and check baseline with default parameters or slightly tuned parameters.
# See my AIAP project for more details. For now I will just define one model.


# baseline = create_model_from_config(cfg.train.create_baseline_model)
# baseline.fit(X_train, y_train)

# y_pred = baseline.predict(X_val)
# accuracy = accuracy_score(y_val, y_pred)
# print(f"Accuracy: {accuracy}")

# y_pred_holdout = baseline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_holdout)
# print(f"Accuracy: {accuracy}")
# time.sleep(100)


# NOTE: optimize.py
partial_train_and_validate_model = functools.partial(
    train_and_validate_model,
    preprocessor=preprocessor,
    logger=logger,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
)
metadata, cfg = optimize(
    cfg=cfg, logger=logger, metadata=metadata, partial=partial_train_and_validate_model
)

# train on best hyperparameters
# NOTE: I did not change X_train to X and y_train to y here, in reality you should.
logger.info(
    "Training model with best hyperparameters..."
    "Updating `X_train` to `X` and `y_train` to `y`"
)
# NOTE: train.py
X = preprocessor.transform(X)
y = label_encoder.transform(y)
# TODO: Uncomment when in production, for legacy reasons I still train the best params
# on X_train and y_train...LOL
# metadata = train_with_best_model_config(cfg, logger, metadata, preprocessor, X=X, y=y)
metadata = train_with_best_model_config(
    cfg, logger, metadata, preprocessor, X=X_train, y=y_train
)


best_model = metadata.model_artifacts["model"]


# NOTE: evaluate.py
X_test_transformed = preprocessor.transform(metadata.X_test_original)
y_test_transformed = label_encoder.transform(metadata.y_test_original)


metadata = predict_on_holdout_set(
    cfg=cfg,
    metadata=metadata,
    logger=logger,
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    run_id=metadata.run_id,
)

metadata = bias_variance(
    cfg,
    metadata,
    logger,
    best_model,
    X_train,
    y_train,
    X_test,
    y_test,
    run_id=metadata.run_id,
)


# NOTE: validate_and_promote.py
# promote.py
client = MlflowClient(tracking_uri=cfg.exp.tracking_uri)
promoter = MLFlowPromotionManager(
    client=client, model_name=cfg.exp.register_model["name"], logger=logger
)

promoter.promote_to_production(metric_name=cfg.train.objective.monitor)
