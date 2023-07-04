import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.base import Connection
from common_utils.core.decorators.timer import timer
from common_utils.core.logger import Logger
from common_utils.data_validator.core import DataFrameValidator
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.data_extraction.extract import Extract
from pipeline_training.data_loading.load import Load
from pipeline_training.data_preprocessing.preprocess import Preprocess
from schema.core import RawSchema, TransformedSchema

# pylint: disable=no-member

cfg = Config()

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

pprint(metadata.raw_df)


# NOTE: start of load.py
storage = GCS(
    project_id=cfg.env.project_id,
    google_application_credentials=cfg.env.google_application_credentials,
    bucket_name=cfg.env.gcs_bucket_name,
)
dvc = SimpleDVC(
    storage=storage,
    remote_bucket_project_name=cfg.env.gcs_bucket_project_name,
    data_dir=cfg.dirs.data.raw,
    metadata_dir=cfg.dirs.stores.blob.raw,
)

load = Load(cfg=cfg, metadata=metadata, logger=logger, dvc=dvc)
metadata = load.run()
pprint(metadata)

expected_raw_schema = RawSchema.to_pd_dtypes()
pprint(expected_raw_schema)

validator = DataFrameValidator(df=metadata.raw_df, schema=expected_raw_schema)
validator.validate_schema().validate_data_types().validate_missing()
# NOTE: at this stage the schema reflected that the data types were not correct
# our number_of_trades is Int64, but it should be int64.

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

# NOTE: preprocess.py
preprocess = Preprocess(cfg=cfg, metadata=metadata, logger=logger)
metadata = preprocess.run()
pprint(metadata.processed_df)
pprint(metadata.processed_df.dtypes)

# TODO: validate again

# NOTE: resampling.py