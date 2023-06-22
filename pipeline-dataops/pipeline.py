from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.base import Connection, Storage
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger
from rich.pretty import pprint
import pandas as pd
from conf.base import Config
from metadata.core import Metadata
from pipeline_dataops.extract.core import from_api, interval_to_milliseconds
from pipeline_dataops.load.core import to_bigquery, to_google_cloud_storage
from pipeline_dataops.schema.core import generate_bq_schema_from_pandas
from pipeline_dataops.transform.core import cast_columns

# pylint: disable=no-member,logging-fstring-interpolation


# TODO:
# 1. Not clean enough because I handle too much logic in the class below.
# 2. If you run elt and fail at transform, the next time you run it, it will
#    have a gap. Try to purposely fail at transform and see what happens.
class Pipeline:
    """Pipeline class that handles the entire ELT process."""

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
        """
        Check if the bucket exists in the storage.

        Returns:
            bool: True if the bucket exists, False otherwise.
        """
        return self.storage.check_if_bucket_exists()

    def dataset_exists(self) -> bool:
        """
        Check if the dataset exists in the connection.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        return self.connection.check_if_dataset_exists()

    def table_exists(self) -> bool:
        """
        Check if the table exists in the connection.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        return self.connection.check_if_table_exists()

    def initial_extract_and_load_and_transform(self) -> Metadata:
        """Do not need to return Metadata since it is an initial extract."""
        if not self.bucket_exists():
            self.storage.create_bucket()

        if not self.dataset_exists():
            self.logger.warning("Dataset does not exist. Creating them now...")
            self.connection.create_dataset()

        assert (
            self.cfg.extract.from_api.start_time is not None
        ), "start_time must be provided to create dataset and table"

        metadata = from_api(
            metadata=self.metadata,
            **self.cfg.extract.from_api.model_dump(mode="python"),
        )
        updated_at = metadata.updated_at
        raw_df = metadata.raw_df
        raw_df["updated_at"] = updated_at

        blob = to_google_cloud_storage(
            df=raw_df,
            gcs=self.storage,
            dataset=self.cfg.env.bigquery_raw_dataset,
            table_name=self.cfg.env.bigquery_raw_table_name,
            updated_at=updated_at,
        )
        self.logger.info(f"File {blob.name} uploaded to {self.storage.bucket_name}")

        schema = generate_bq_schema_from_pandas(raw_df)

        self.connection.create_table(schema=schema)

        # NOTE: We hardcode the write_disposition to WRITE_APPEND
        # as we will use this operation to update the table.
        to_bigquery(
            df=raw_df,
            bq=self.connection,
            schema=schema,
            write_disposition="WRITE_APPEND",
        )

        # TODO: The below is repeated step for the normal elt as well.
        transformed_df = cast_columns(
            df=raw_df, **self.cfg.transform.cast_columns.model_dump(mode="python")
        )
        blob = to_google_cloud_storage(
            transformed_df,
            gcs=self.storage,
            dataset=self.cfg.env.bigquery_transformed_dataset,
            table_name=self.cfg.env.bigquery_transformed_table_name,
            updated_at=updated_at,
        )
        schema = generate_bq_schema_from_pandas(transformed_df)
        self.connection.table_name = self.cfg.env.bigquery_transformed_table_name
        self.connection.create_table(schema=schema)
        to_bigquery(
            df=transformed_df,
            bq=self.connection,
            write_disposition="WRITE_APPEND",
        )
        return metadata

    def is_first_run(self) -> bool:
        return not self.dataset_exists() or not self.table_exists()

    @property
    def max_open_time_query(self) -> str:
        return f"""
        SELECT MAX(open_time) as max_open_time
        FROM `{bq.table_id}`
        """

    def extract(self) -> Metadata:
        """This method assumes that the dataset and table already exist and they
        are already populated with data. This method will only extract the data
        from the maximum open_date onwards."""
        self.logger.info("Extracting data from API")

        # Query to find the maximum open_date
        query = self.max_open_time_query
        max_date_result: pd.DataFrame = bq.query(query, as_dataframe=True)
        max_open_time: int = max(max_date_result["max_open_time"])

        # now max_open_time is your new start_time
        start_time = max_open_time + interval_to_milliseconds(
            interval=self.cfg.extract.from_api.interval
        )
        self.logger.warning("Overwriting `start_time` in the config.")
        # NOTE: We are overwriting the start_time here.
        self.cfg.extract.from_api.start_time = int(start_time)

        metadata = from_api(
            metadata=self.metadata,
            **self.cfg.extract.from_api.model_dump(mode="python"),
        )
        return metadata

    def load(self, metadata: Metadata) -> Metadata:
        raw_df = metadata.raw_df
        # call it so that it persists across the next few lines
        updated_at = metadata.updated_at
        raw_df["updated_at"] = updated_at

        blob = to_google_cloud_storage(
            df=raw_df,
            gcs=self.storage,
            dataset=self.cfg.env.bigquery_raw_dataset,
            table_name=self.cfg.env.bigquery_raw_table_name,
            updated_at=updated_at,
        )
        self.logger.info(f"File {blob.name} uploaded to {self.storage.bucket_name}.")

        # Append the new data to the existing table
        to_bigquery(raw_df, bq=self.connection, write_disposition="WRITE_APPEND")
        return metadata

    def transform(self, metadata: Metadata) -> Metadata:
        transformed_df = cast_columns(
            df=metadata.raw_df,
            **self.cfg.transform.cast_columns.model_dump(mode="python"),
        )
        blob = to_google_cloud_storage(
            df=transformed_df,
            gcs=self.storage,
            dataset=self.cfg.env.bigquery_transformed_dataset,
            table_name=self.cfg.env.bigquery_transformed_table_name,
            updated_at=metadata.updated_at,
        )
        self.logger.info(f"File {blob.name} uploaded to {self.storage.bucket_name}.")
        schema = generate_bq_schema_from_pandas(transformed_df)
        self.connection.table_name = self.cfg.env.bigquery_transformed_table_name
        to_bigquery(
            df=transformed_df,
            bq=self.connection,
            schema=schema,
            write_disposition="WRITE_APPEND",
        )
        metadata_dict = {"transformed_df": transformed_df}
        metadata.set_attrs(metadata_dict)
        return metadata

    def run(self) -> Metadata:
        if self.is_first_run():
            self.logger.info("First run detected. Running initial extract and load")
            metadata = self.initial_extract_and_load_and_transform()
        else:
            metadata = self.extract()
            pprint(metadata.raw_df)
            logger.info(
                "Extracted data from API, proceeding to load to BigQuery and GCS."
            )

            metadata = self.load(metadata)
            metadata = self.transform(metadata)

        return metadata


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
    pipeline.run()
