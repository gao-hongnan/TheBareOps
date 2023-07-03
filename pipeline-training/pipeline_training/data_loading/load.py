import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.base import Connection
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from rich.pretty import pprint

from conf.base import RUN_ID, Config
from conf.directory.base import ROOT_DIR
from metadata.core import Metadata


def load(
    cfg: Config,
    metadata: Metadata,
    logger: Logger,
    dvc: Optional[SimpleDVC] = None,
) -> Metadata:
    """
    This function loads data from the metadata into a CSV file, and
    optionally tracks that file using Data Version Control (DVC).

    Design Principles:
    ------------------
    1. Single Responsibility Principle (SRP): This function solely
       handles loading data and potentially tracking it with DVC.

    2. Dependency Inversion Principle (DIP): Dependencies (logger,
       directories, metadata, and optionally DVC) are passed as
       arguments for better testability and flexibility.

    3. Open-Closed Principle (OCP): The function supports optional DVC
       tracking, showing it can be extended without modifying the
       existing code. In other words, the function is open for
       extension because it can be extended to support DVC tracking,
       and closed for modification because the existing code does not
       need to be modified to support DVC tracking.

    4. Use of Type Hints: Type hints are used consistently for clarity
       and to catch type-related errors early.

    5. Logging: Effective use of logging provides transparency and aids
       in troubleshooting.


    Areas of Improvement:
    ---------------------
    1. Exception Handling: More specific exception handling could
       improve error management.

    2. Liskov Substitution Principle (LSP): Using interfaces or base
       classes for the inputs could enhance flexibility and adherence
       to LSP.

    3. Encapsulation: Consider if direct manipulation of `metadata`
       attributes should be encapsulated within `metadata` methods.

    Parameters
    ----------
    metadata: Metadata
        The Metadata object containing the data to be loaded.
    logger: Logger
        The Logger object for logging information and errors.
    dirs: Directories
        The Directories object with the directories for data loading.
    dvc: Optional[SimpleDVC]
        The optional DVC object for data file tracking.

    Returns
    -------
    Metadata
        The Metadata object with updated information.
    """
    logger.info("Reading data from metadata computed in the previous step...")
    raw_df = metadata.raw_df
    table_name = metadata.raw_table_name
    filepath: Path = cfg.dirs.data.raw / f"{table_name}.csv"

    raw_df.to_csv(filepath, index=False)

    if dvc is not None:
        # add local file to dvc
        raw_dvc_metadata = dvc.add(filepath)
        try:
            dvc.push(filepath)
        except Exception as error:  # pylint: disable=broad-except
            logger.error(f"File is already tracked by DVC. Error: {error}")

    attr_dict = {
        "raw_file_size": filepath.stat().st_size,
        "raw_file_format": filepath.suffix[1:],
        "raw_dvc_metadata": raw_dvc_metadata if dvc is not None else None,
    }
    metadata.set_attrs(attr_dict)
    return metadata


if __name__ == "__main__":
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
        dataset=cfg.env.bigquery_transformed_dataset,
        table_name=cfg.env.bigquery_transformed_table_name,
    )

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

    from pipeline_training.data_extraction.extract import (
        test_extract_from_data_warehouse,
    )

    ## extract data from data warehouse
    metadata = test_extract_from_data_warehouse(
        cfg=cfg, metadata=metadata, logger=logger, connection=connection
    )

    ## load data
    metadata = load(cfg=cfg, metadata=metadata, logger=logger, dvc=dvc)
    pprint(metadata)
