from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Type

import rich
from common_utils.core.common import load_env_vars
from pydantic import BaseModel, Field

from conf.directory.base import ROOT_DIR


def is_docker() -> Literal[True, False]:
    """
    Checks if the environment is running inside a Docker container.

    Returns
    -------
    Literal[True, False]
        Returns True if the environment is Docker, otherwise False.
    """
    path = "/.dockerenv"
    return os.path.exists(path)


def is_kubernetes() -> Literal[True, False]:
    """
    Checks if the environment is running inside a Kubernetes cluster.

    Returns
    -------
    Literal[True, False]
        Returns True if the environment is Kubernetes, otherwise False.
    """
    return os.getenv("KUBERNETES_SERVICE_HOST") is not None


class Environment(BaseModel):
    """
    Pydantic model for Environment variables.

    Attributes
    ----------
    project_id : Optional[str]
        Google Cloud Project ID.
    google_application_credentials : Optional[str]
        Google Application Credentials.
    gcs_bucket_name : Optional[str]
        Google Cloud Storage Bucket Name.
    gcs_bucket_project_name : Optional[str]
        Google Cloud Storage Bucket Project Name.
    bigquery_raw_dataset : Optional[str]
        BigQuery Raw Dataset.
    bigquery_raw_table_name : Optional[str]
        BigQuery Raw Table Name.
    bigquery_transformed_dataset : Optional[str]
        BigQuery Processed Dataset.
    bigquery_transformed_table_name : Optional[str]
        BigQuery Processed Table Name.
    """

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

    bigquery_transformed_dataset: Optional[str] = Field(
        default=os.getenv("BIGQUERY_TRANSFORMED_DATASET"),
        description="BigQuery Processed Dataset",
    )

    bigquery_transformed_table_name: Optional[str] = Field(
        default=os.getenv("BIGQUERY_TRANSFORMED_TABLE_NAME"),
        description="BigQuery Processed Table Name",
    )

    @classmethod
    def create_instance(cls: Type[Environment]) -> Environment:
        if not is_docker() and not is_kubernetes():
            # TODO: Consider using the same logger instead of rich.
            rich.print("Not running inside docker")
            load_env_vars(root_dir=ROOT_DIR)

        # NOTE: This assumes that all the env variables defined in
        # this class are environment variables. This is a hackish way
        # because we assume each attribute's upper case name is the
        # environment variable name.
        env_vars: Dict[str, str] = {
            name: os.getenv(name.upper()) for name in cls.__annotations__.keys()
        }
        return cls(**env_vars)

    class Config:
        """Configuration for Pydantic model."""

        frozen: bool = True


if __name__ == "__main__":
    env = Environment.create_instance()
    rich.print(env)
