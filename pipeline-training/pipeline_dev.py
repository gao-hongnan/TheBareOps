"""Pipeline without pipeline lol."""
import functools
import logging
import time

import numpy as np
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.logger import Logger
from common_utils.data_validator.core import DataFrameValidator
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.data_cleaning.clean import Clean
from pipeline_training.data_extraction.extract import Extract
from pipeline_training.data_loading.load import Load
from pipeline_training.data_resampling.resample import Resampler
from pipeline_training.model_training.optimize import optimize
from pipeline_training.model_training.preprocess import Preprocessor
from pipeline_training.model_training.train import (
    create_model_from_config,
    train_and_validate_model,
    train_with_best_model_config,
)
from common_utils.tests.core import compare_test_case, compare_test_cases
from pipeline_training.utils.common import log_data_splits_summary
from schema.core import CleanedSchema, RawSchema

# pylint: disable=no-member

# FIXME: Ask how to modify logger to capture the logs from external modules.

cfg = Config()

logger = Logger(
    log_file="pipeline_training.log",
    log_root_dir=cfg.dirs.stores.logs,
    module_name=__name__,
    propagate=False,
    level=logging.DEBUG,
).logger

compare_test_case(
    actual=cfg.extract.extract_from_data_warehouse.query,
    expected="""
    SELECT *
    FROM `gao-hongnan.thebareops_production.raw_binance_btcusdt_spot` t
    WHERE t.utc_datetime > DATETIME(TIMESTAMP "2023-06-09 00:00:00 UTC")
    ORDER BY t.utc_datetime DESC
    LIMIT 3000;
    """,
    description="ExtractFromDataWarehouse.query",
    logger=logger,
)

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
compare_test_cases(
    actual_list=[metadata.raw_dataset, metadata.raw_table_name],
    expected_list=["thebareops_production", "raw_binance_btcusdt_spot"],
    description_list=["raw_dataset", "raw_table_name"],
    logger=logger,
)

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
compare_test_cases(
    actual_list=[
        metadata.raw_dvc_metadata["filename"],
        metadata.raw_dvc_metadata["md5"],
        metadata.raw_dvc_metadata["remote_dvc_dir_name"],
    ],
    expected_list=[
        "raw_binance_btcusdt_spot.csv",
        "0d0d50a7f8983eed7b6e2439e26187a0",
        "gaohn-dvc",
    ],
    description_list=["raw_dvc filename", "raw_dvc md5", "raw_dvc remote_dvc_dir_name"],
    logger=logger,
)

# reinitialize to pull
# cfg = initialize_project(ROOT_DIR)
# gcs = GCS(
#     project_id=cfg.env.project_id,
#     google_application_credentials=cfg.env.google_application_credentials,
#     bucket_name=cfg.env.gcs_bucket_name,
# )
# dvc = SimpleDVC(data_dir=cfg.general.dirs.data.raw, storage=gcs)
# filename = "filtered_movies_incremental.csv"
# remote_project_name = "imdb"
# dvc.pull(filename=filename, remote_project_name=remote_project_name)

# NOTE: clean.py
clean_dvc = SimpleDVC(
    storage=storage,
    remote_bucket_project_name=cfg.env.gcs_bucket_project_name,
    data_dir=cfg.dirs.data.processed,
    metadata_dir=cfg.dirs.stores.blob.processed,
)
clean = Clean(cfg=cfg, metadata=metadata, logger=logger, dvc=clean_dvc)
metadata = clean.run()
compare_test_cases(
    actual_list=[
        metadata.processed_dvc_metadata["filename"],
        metadata.processed_dvc_metadata["md5"],
        metadata.processed_dvc_metadata["remote_dvc_dir_name"],
        metadata.feature_columns,
        metadata.target_columns,
        metadata.raw_df,
        # metadata.processed_df.dtypes.to_dict(),
    ],
    expected_list=[
        "processed_binance_btcusdt_spot.csv",
        "5b4b835b25fc56bf87fa7a48617a8624",
        "gaohn-dvc",
        [
            "open",
            "high",
            "low",
            "volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ],
        "price_increase",
        None,
    ],
    description_list=[
        "processed_dvc filename",
        "processed_dvc md5",
        "processed_dvc remote_dvc_dir_name",
        "Number of columns in cleaned dataframe",
        "feature_columns",
        "target_columns",
        "Check if raw_df is released",
        # "Check if processed_df dtypes are correct",
    ],
    logger=logger,
)

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
compare_test_cases(
    actual_list=[
        metadata.X_train,
        metadata.X_val,
        metadata.X_test,
        metadata.y_train,
        metadata.y_val,
        metadata.y_test,
    ],
    expected_list=[
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ],
    description_list=[
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test",
    ],
    logger=logger,
)
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
    }
)
compare_test_cases(
    actual_list=[metadata.classes, metadata.num_classes],
    expected_list=[np.array([0, 1]), 2],
    description_list=["num_classes", "classes"],
    logger=logger,
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
# time.sleep(10000)
# train on best hyperparameters
# NOTE: I did not change X_train to X and y_train to y here, in reality you should.
logger.info(
    "Training model with best hyperparameters..."
    "Updating `X_train` to `X` and `y_train` to `y`"
)


metadata = train_with_best_model_config(cfg, logger, metadata, preprocessor, X=X, y=y)
compare_test_cases(
    actual_list=[metadata.best_params, metadata.model_artifacts["model_config"]],
    expected_list=[
        {
            "model__alpha": 0.00018152669990587287,
            "model__power_t": 0.2685947407265318,
            "best_trial": 0,
        },
        {
            "alpha": 0.00018152669990587287,
            "average": False,
            "class_weight": None,
            "early_stopping": False,
            "epsilon": 0.1,
            "eta0": 0.1,
            "fit_intercept": True,
            "l1_ratio": 0.15,
            "learning_rate": "optimal",
            "loss": "log_loss",
            "max_iter": 10,
            "n_iter_no_change": 5,
            "n_jobs": None,
            "penalty": "l2",
            "power_t": 0.2685947407265318,
            "random_state": 1992,
            "shuffle": True,
            "tol": 0.001,
            "validation_fraction": 0.1,
            "verbose": 0,
            "warm_start": True,
        },
    ],
    description_list=["best_params"],
    logger=logger,
)
pprint(metadata)

best_model = create_model_from_config(cfg.train.create_model)
best_model.fit(X_train, y_train)

y_pred_holdout = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_holdout)
print(f"Accuracy: {accuracy}")
assert accuracy == 0.8155555555555556


# NOTE:
# BASELINE MODEL
# OPTIMIZE ON BASELINE MODEL
# TRAIN MODEL WITH OPTIMIZED PARAMETERS
# EVALUATE MODEL
# VALIDATE MODEL
# PUSH MODEL TO PRODUCTION IF SATISFACTORY
# SERVE
# MONITOR
# CICD Trigger training -> Trigger Serving of Image


# @timer(display_table=True) for the full pipeline?
