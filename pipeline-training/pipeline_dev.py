"""Pipeline without pipeline lol."""
import logging
import time

from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.logger import Logger
from common_utils.data_validator.core import DataFrameValidator
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint
from sklearn.metrics import accuracy_score

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.data_cleaning.clean import Clean
from pipeline_training.data_extraction.extract import Extract
from pipeline_training.data_loading.load import Load
from pipeline_training.data_resampling.resampler import Resampler
from pipeline_training.model_training.optimize import optimize
from pipeline_training.model_training.preprocess import Preprocessor
from pipeline_training.model_training.train import (
    Trainer,
    create_baseline_model,
    create_model,
)
from pipeline_training.utils.common import (
    compare_test_case,
    compare_test_cases,
    log_data_splits_summary,
)
from schema.core import RawSchema

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
        metadata.processed_df.dtypes.to_dict(),
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
        {
            "utc_datetime": "datetime64[ns]",
            "open_time": "int",
            "open": "float",
            "high": "float",
            "low": "float",
            "close": "float",
            "volume": "float",
            "close_time": "int",
            "quote_asset_volume": "float",
            "number_of_trades": "int",
            "taker_buy_base_asset_volume": "float",
            "taker_buy_quote_asset_volume": "float",
            "ignore": "str",
            "updated_at": "datetime64[ns]",
        },
    ],
    description_list=[
        "processed_dvc filename",
        "processed_dvc md5",
        "processed_dvc remote_dvc_dir_name",
        "feature_columns",
        "target_columns",
        "Check if raw_df is released",
        "Check if processed_df dtypes are correct",
    ],
    logger=logger,
)


# pprint(metadata.processed_df.dtypes)
time.sleep(10000)

# TODO: validate again after preprocessing

# TODO: Make resample a class and log data splits as method.
# NOTE: resampling.py
# TODO: Consider remove X and y from init of Resampler since it can be obtained
# from metadata.
X = metadata.X
y = metadata.y

resampler = Resampler(cfg=cfg, metadata=metadata, logger=logger, X=X, y=y)
metadata = resampler.metadata

# X_train, X_val, y_train, y_val = (
#     metadata.X_train,
#     metadata.X_val,
#     metadata.y_train,
#     metadata.y_val,
# )


# NOTE: Subsetting resampler works why? Because of __getitem__ method.
X_train, y_train = resampler["train"]
X_val, y_val = resampler["val"]
X_test, y_test = resampler["test"]

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

# here x_train, x_val, x_test are numpy arrays cause imputer returns numpy arrays
# and therefore it.
metadata.set_attrs({"X_train": X_train, "X_val": X_val, "X_test": X_test})

# TODO: Technically, BASELINE MODEL can be rule based or something simple for
# ml models to beat. Check https://github.com/gao-hongnan/aiap-batch10-coronary-artery-disease/tree/master/notebooks
# for more details.
# So the idea is before the pipeline run in production

# In reality, this step can select like a list of [model1, model2, model3]
# and check baseline with default parameters or slightly tuned parameters.
# See my AIAP project for more details. For now I will just define one model.


baseline = create_baseline_model(cfg)
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

y_pred_holdout = baseline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_holdout)
print(f"Accuracy: {accuracy}")


# trainer = Trainer(cfg=cfg, metadata=metadata, logger=logger, preprocessor=preprocessor)
# metadata = trainer.train_model(trial=None)
# metadata = train_model(
#     cfg=cfg, logger=logger, metadata=metadata, model=model, trial=None
# )

# optimize.py
metadata, cfg = optimize(
    cfg=cfg, metadata=metadata, logger=logger, preprocessor=preprocessor
)

# train on best hyperparameters
logger.info(
    "Training model with best hyperparameters...Updating `X_train` to `X` and `y_train` to `y`"
)
# TODO: but i did not for simplicity sake.
# X = preprocessor.transform(X)
# metadata.X_train = X
# metadata.y_train = y.to_numpy()

trainer = Trainer(cfg=cfg, metadata=metadata, logger=logger, preprocessor=preprocessor)
metadata = trainer.train()
pprint(metadata)


best_model = create_model(cfg)
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
