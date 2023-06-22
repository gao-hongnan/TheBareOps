from __future__ import annotations

import os
import unittest
from typing import Literal, Optional, Type

import rich
from common_utils.core.common import load_env_vars
from pydantic import BaseModel, Field

from conf.directory.base import ROOT_DIR


def is_docker() -> Literal[True, False]:
    path = "/.dockerenv"
    return os.path.exists(path)


class Environment(BaseModel):
    project_id: Optional[str] = Field(
        default=os.getenv("PROJECT_ID"), description="Google Cloud Project ID"
    )
    google_application_credentials: Optional[str] = Field(
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        description="Google Application Credentials",
    )
    gcs_bucket_name: Optional[str] = Field(
        default=os.getenv("GCS_BUCKET_NAME"),
        description="Google Cloud Storage Bucket Name",
    )
    gcs_bucket_project_name: Optional[str] = Field(
        default=os.getenv("GCS_BUCKET_PROJECT_NAME"),
        description="Google Cloud Storage Bucket Project Name",
    )
    bigquery_raw_dataset: Optional[str] = Field(
        default=os.getenv("BIGQUERY_RAW_DATASET"), description="BigQuery Raw Dataset"
    )
    bigquery_raw_table_name: Optional[str] = Field(
        default=os.getenv("BIGQUERY_RAW_TABLE_NAME"),
        description="BigQuery Raw Table Name",
    )

    bigquery_processed_dataset: Optional[str] = Field(
        default=os.getenv("BIGQUERY_PROCESSED_DATASET"),
        description="BigQuery Processed Dataset",
    )

    bigquery_processed_table_name: Optional[str] = Field(
        default=os.getenv("BIGQUERY_PROCESSED_TABLE_NAME"),
        description="BigQuery Processed Table Name",
    )

    @classmethod
    def create_instance(cls: Type[Environment]) -> Environment:
        if not is_docker():
            rich.print("Not running inside docker")
            load_env_vars(root_dir=ROOT_DIR)
        return cls(
            project_id=os.getenv("PROJECT_ID"),
            google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            gcs_bucket_name=os.getenv("GCS_BUCKET_NAME"),
            gcs_bucket_project_name=os.getenv("GCS_BUCKET_PROJECT_NAME"),
            bigquery_raw_dataset=os.getenv("BIGQUERY_RAW_DATASET"),
            bigquery_raw_table_name=os.getenv("BIGQUERY_RAW_TABLE_NAME"),
            bigquery_processed_dataset=os.getenv("BIGQUERY_PROCESSED_DATASET"),
            bigquery_processed_table_name=os.getenv("BIGQUERY_PROCESSED_TABLE_NAME"),
        )

    class Config:
        frozen: bool = True


if __name__ == "__main__":
    env = Environment.create_instance()
    rich.print(env)

    # # raise error test that it can be mutated with frozen
    # with unittest.TestCase.assertRaises(unittest.TestCase, AttributeError):
    #     env.project_id = "test"
