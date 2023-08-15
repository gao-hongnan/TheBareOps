"""Pipeline class that handles the entire ELT process. It can be treated as a DAG."""
import logging
import pickle
import time

import pandas as pd
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.base import Connection, Storage
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger

from conf.base import RUN_ID, Config
from conf.directory.base import ROOT_DIR
from metadata.core import Metadata
from pipeline_dataops.extract.core import from_api, interval_to_milliseconds
from pipeline_dataops.load.core import to_bigquery, to_google_cloud_storage
from pipeline_dataops.transform.core import cast_columns
from pipeline_dataops.validate.core import validate_non_null_columns, validate_schema
from schema.core import RawSchema, TransformedSchema

# pylint: disable=no-member,logging-fstring-interpolation,invalid-name


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
        """
        Initialize the pipeline with the provided configurations, metadata,
        logger, connection, and storage.

        Parameters
        ----------
        cfg : Config
            The configuration object containing all the necessary parameters.
        metadata : Metadata
            The metadata object used to store metadata about the pipeline.
        logger : Logger
            The logger object used for logging.
        connection : Connection
            The connection object used for database interactions. In our example
            here, it is BigQuery.
        storage : Storage
            The storage object used for storing files. In our example here, it is
            Google Cloud Storage.
        """
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

    def validate_raw(self, df: pd.DataFrame) -> None:
        """
        Validate the raw data against a predefined schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw data to validate.
        """
        self.logger.info("Validating raw schema...")
        validate_schema(logger=self.logger, df=df, validator=RawSchema)
        self.logger.info("✅ Raw schema is valid.")

        self.logger.info("Validating non-null columns...")
        validate_non_null_columns(
            logger=self.logger, df=df, columns=list(RawSchema.__annotations__.keys())
        )
        self.logger.info("✅ There are no null values in the specified columns.")

    def validate_transformed(self, df: pd.DataFrame) -> None:
        """
        Validate the transformed data against a predefined schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with transformed data to validate.
        """
        self.logger.info("Validating transformed schema...")
        validate_schema(logger=self.logger, df=df, validator=TransformedSchema)
        self.logger.info("✅ Transformed schema is valid.")

        self.logger.info("Validating non-null columns...")
        validate_non_null_columns(
            logger=self.logger,
            df=df,
            columns=list(TransformedSchema.__annotations__.keys()),
        )
        self.logger.info("✅ There are no null values in the specified columns.")

    def initial_extract_and_load_and_transform(self) -> Metadata:
        """
        Execute initial extract, load, and transform operations. Assumes
        it is the first run.

        Returns
        -------
        Metadata
            Updated metadata after operations.
        """
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

        blob = to_google_cloud_storage(
            df=raw_df,
            gcs=self.storage,
            dataset=self.cfg.env.bigquery_raw_dataset,
            table_name=self.cfg.env.bigquery_raw_table_name,
            updated_at=updated_at,
        )
        self.logger.info(f"File {blob.name} uploaded to {self.storage.bucket_name}")

        self.validate_raw(df=raw_df)
        schema = RawSchema.to_bq_schema()

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
        self.validate_transformed(df=transformed_df)

        blob = to_google_cloud_storage(
            transformed_df,
            gcs=self.storage,
            dataset=self.cfg.env.bigquery_transformed_dataset,
            table_name=self.cfg.env.bigquery_transformed_table_name,
            updated_at=updated_at,
        )
        schema = TransformedSchema.to_bq_schema()
        self.connection.table_name = self.cfg.env.bigquery_transformed_table_name
        self.connection.create_table(schema=schema)
        to_bigquery(
            df=transformed_df,
            bq=self.connection,
            write_disposition="WRITE_APPEND",
        )
        return metadata

    def is_first_run(self) -> bool:
        """
        Check if the current run is the first by checking if the connection's
        dataset and table exists.

        Returns
        -------
        bool
            True if it's the first run, False otherwise.
        """
        return not self.dataset_exists() or not self.table_exists()

    @property
    def max_open_time_query(self) -> str:
        """
        SQL Query to fetch the maximum open_date.

        Returns
        -------
        str
            String SQL query.
        """
        return f"""
        SELECT MAX(open_time) as max_open_time
        FROM `{bq.table_id}`
        """

    def extract(self) -> Metadata:
        """
        Extract data from the API from the maximum open_date onwards.
        Assumes dataset and table already exist and populated.

        Returns
        -------
        Metadata
            Updated metadata after the extraction.
        """
        self.logger.info("Extracting data from API")

        # Query to find the maximum open_date
        query = self.max_open_time_query
        max_date_result: pd.DataFrame = bq.query(query, as_dataframe=True)
        max_open_time: int = max(max_date_result["max_open_time"])

        # now max_open_time is your new start_time, reason to add
        # interval_to_milliseconds is to avoid duplicates.
        # meaning, if you have max_open_time to be 1000, and interval is 1m,
        # then the start_time should be 1000 + 1 = 1001 and not 1000.
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
        """
        Load data to Google Cloud Storage and BigQuery.

        Parameters
        ----------
        metadata : Metadata
            Updated metadata after extraction operation.

        Returns
        -------
        Metadata
            Updated metadata after the load operation.
        """
        raw_df = metadata.raw_df

        # define it here so that it persists across the next few lines
        updated_at = metadata.updated_at

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
        """
        Perform the data transformation and load to Google Cloud Storage and BigQuery.

        Parameters
        ----------
        metadata : Metadata
            Metadata with raw extracted data.

        Returns
        -------
        Metadata
            Updated metadata with transformed data.
        """
        # TODO: consider update updated_at in transform instead of using
        # the updated_at from the extract.
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
        schema = TransformedSchema.to_bq_schema()

        self.logger.warning(
            "Overwriting `table_name` in the config to transformed table."
        )
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

    def upload_metadata_to_artifacts(self, metadata: Metadata) -> None:
        """Upload the metadata object to the artifacts directory locally and
        then upload stores directory (artifacts is a subdirectory of stores) to
        Storage."""
        # Serialize the metadata object
        local_metadata_path = f"{self.cfg.dirs.stores.artifacts}/metadata.pkl"
        with open(local_metadata_path, "wb") as file:
            pickle.dump(metadata, file)

    def upload_config_to_artifacts(self, config: Config) -> None:
        """Upload the config object to the artifacts directory locally and
        then upload stores directory (artifacts is a subdirectory of stores) to
        Storage."""
        # Serialize the config object
        local_config_path = f"{self.cfg.dirs.stores.artifacts}/config.pkl"
        with open(local_config_path, "wb") as file:
            pickle.dump(config, file)

    def upload_stores(self) -> None:
        """Upload the stores directory to Storage."""
        self.storage.upload_directory(
            source_dir=str(self.cfg.dirs.stores.base),
            destination_dir=(
                f"{self.cfg.env.gcs_bucket_project_name}/"
                f"{self.cfg.general.pipeline_name}/{RUN_ID}"
            ),
        )

    def run(self) -> Metadata:
        """
        Execute the entire pipeline, which includes extraction,
        validation, loading, transformation, and updating metadata.


        Returns
        -------
        Metadata
            The updated metadata after running the pipeline.
        """
        start_time = time.time()
        if self.is_first_run():
            self.logger.info("First run detected. Running initial extract and load")
            metadata = self.initial_extract_and_load_and_transform()
        else:
            metadata = self.extract()
            self.logger.info(
                "Extracted data from API, now validating data schema and skews."
            )

            self.validate_raw(df=metadata.raw_df)
            self.logger.info(
                "Validation successful. Now loading data to Google Cloud Storage and BigQuery."
            )

            metadata = self.load(metadata)
            self.logger.info("Loading successful. Now transforming data.")

            metadata = self.transform(metadata)
            self.logger.info(
                "Transformation successful. Now validating transformed data."
            )

            self.validate_transformed(df=metadata.transformed_df)
            self.logger.info("Validation successful. Now updating metadata.")

        end_time = time.time()
        pipeline_time_taken = end_time - start_time
        metadata.set_attrs({"pipeline_time_taken": pipeline_time_taken})
        self.logger.info("Pipeline run successful. Time taken: %s", pipeline_time_taken)

        self.logger.info("Uploading config to local artifacts directory.")
        self.upload_config_to_artifacts(config=self.cfg)

        self.logger.info("Uploading metadata to local artifacts directory.")
        self.upload_metadata_to_artifacts(metadata=metadata)

        self.logger.info("Uploading stores directory to Storage.")
        self.upload_stores()

        return metadata


if __name__ == "__main__":
    cfg = Config()
    seed_all(cfg.general.seed)

    metadata = Metadata()

    logger = Logger(
        log_file="pipeline_training.log",
        log_root_dir=cfg.dirs.stores.logs,
        module_name=__name__,
        propagate=False,
        level=logging.DEBUG,
    ).logger

    # log run_id and root_dir
    logger.info(f"run_id: {RUN_ID}\nroot_dir: {ROOT_DIR}")

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
