from typing import List, Optional

import pandas as pd
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.base import Connection, Storage
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger
from google.cloud import bigquery, storage
from rich.pretty import pprint

from conf.base import Config
from metadata.core import Metadata
from pipeline_dataops.extract.core import from_api

# pylint: disable=no-member


def generate_bq_schema_from_pandas(df: pd.DataFrame) -> List[bigquery.SchemaField]:
    """
    Convert pandas dtypes to BigQuery dtypes.

    Parameters
    ----------
    dtypes : pandas Series
        The pandas dtypes to convert.

    Returns
    -------
    List[google.cloud.bigquery.SchemaField]
        The corresponding BigQuery dtypes.
    """
    dtype_mapping = {
        "int64": bigquery.enums.SqlTypeNames.INT64,
        "float64": bigquery.enums.SqlTypeNames.FLOAT64,
        "object": bigquery.enums.SqlTypeNames.STRING,
        "bool": bigquery.enums.SqlTypeNames.BOOL,
        "datetime64[ns]": bigquery.enums.SqlTypeNames.DATETIME,
        "datetime64[ns, Asia/Singapore]": bigquery.enums.SqlTypeNames.DATETIME,
    }

    schema = []

    for column, dtype in df.dtypes.items():
        if str(dtype) not in dtype_mapping:
            raise ValueError(f"Cannot convert {dtype} to a BigQuery data type.")

        bq_dtype = dtype_mapping[str(dtype)]
        field = bigquery.SchemaField(name=column, field_type=bq_dtype, mode="NULLABLE")
        schema.append(field)

    return schema


def to_bigquery(
    df: pd.DataFrame,
    bq: BigQuery,
    write_disposition: str = "WRITE_APPEND",
    schema: Optional[List[bigquery.SchemaField]] = None,
) -> None:
    job_config = bq.load_job_config(schema=schema, write_disposition=write_disposition)
    bq.load_table_from_dataframe(df=df, job_config=job_config)


def to_google_cloud_storage(
    df: pd.DataFrame, gcs: GCS, dataset: str, table_name: str, updated_at: str
) -> storage.Blob:
    blob = gcs.create_blob(f"{dataset}/{table_name}/{updated_at}.csv")
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
    return blob


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        metadata: Metadata,
        logger: Logger,
        connection: Connection,
        storage: Storage,
    ) -> None:
        self.cfg = cfg
        self.metadata = metadata
        self.logger = logger
        self.connection = connection
        self.storage = storage

    def bucket_exists(self) -> bool:
        return self.storage.check_if_bucket_exists()

    def dataset_exists(self) -> bool:
        return self.connection.check_if_dataset_exists()

    def table_exists(self) -> bool:
        return self.connection.check_if_table_exists()

    def extract(self) -> Metadata:
        self.logger.info("Extracting data from API")

        if not self.bucket_exists():
            self.storage.create_bucket()

        if not self.dataset_exists() or not self.table_exists():
            self.logger.warning("Dataset or table does not exist. Creating them now...")
            assert (
                self.cfg.extract.from_api.start_time is not None
            ), "start_time must be provided to create dataset and table"

            # NOTE: Be careful, here we are overwriting the metadata object.
            self.metadata = from_api(
                metadata=self.metadata,
                **self.cfg.extract.from_api.model_dump(mode="python"),
            )
            raw_df = self.metadata.raw_df

            blob = to_google_cloud_storage(
                df=raw_df,
                gcs=self.storage,
                dataset=self.cfg.env.bigquery_raw_dataset,
                table_name=self.cfg.env.bigquery_raw_table_name,
                updated_at=self.metadata.updated_at,
            )
            self.logger.info(f"File {blob.name} uploaded to {self.storage.bucket_name}")

            schema = generate_bq_schema_from_pandas(raw_df)
            pprint(schema)

            if not self.dataset_exists():
                self.connection.create_dataset()

            if not self.table_exists():
                self.connection.create_table(schema=schema)

            # TODO: ALL TO CONFIG
            to_bigquery(
                df=raw_df,
                bq=self.connection,
                schema=schema,
                write_disposition="WRITE_TRUNCATE",
            )
        return self.metadata


if __name__ == "__main__":
    cfg = Config()
    # pprint(cfg)
    seed_all(cfg.general.seed)

    metadata = Metadata()
    # pprint(metadata)

    logger = Logger(
        log_file="pipeline_training.log",
        log_root_dir=cfg.dirs.stores.logs,
        module_name=__name__,
        propagate=False,
    ).logger

    gcs = GCS(
        project_id=cfg.env.project_id,
        google_application_credentials=cfg.env.google_application_credentials,
        bucket_name=cfg.env.gcs_bucket_name,
    )

    bq = BigQuery(
        project_id=cfg.env.project_id,
        google_application_credentials=cfg.env.google_application_credentials,
        dataset=cfg.env.bigquery_raw_dataset,
        table_name=cfg.env.bigquery_raw_table_name,
    )
    pipeline = Pipeline(
        cfg=cfg, metadata=metadata, logger=logger, connection=bq, storage=gcs
    )
    pipeline.extract()
