from typing import List, Optional

import pandas as pd
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from google.cloud import bigquery, storage


def to_bigquery(
    df: pd.DataFrame,
    bq: BigQuery,
    write_disposition: str = "WRITE_APPEND",
    schema: Optional[List[bigquery.SchemaField]] = None,
) -> None:
    """
    Load data from a pandas DataFrame to a BigQuery table.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to load data from.
    bq : BigQuery
        The BigQuery client to use for the upload.
    write_disposition : str, optional
        Write disposition specifies the action if the table already exists.
        The default value is "WRITE_APPEND" which will append the data to the table.
        Other possible values are "WRITE_EMPTY" and "WRITE_TRUNCATE".
    schema : List[bigquery.SchemaField], optional
        The schema to be used for the BigQuery table.
        If not provided, the schema is inferred from the DataFrame.

    Returns
    -------
    None
    """
    job_config = bq.load_job_config(schema=schema, write_disposition=write_disposition)
    bq.load_table_from_dataframe(df=df, job_config=job_config)


def to_google_cloud_storage(
    df: pd.DataFrame, gcs: GCS, dataset: str, table_name: str, updated_at: str
) -> storage.Blob:
    """
    Upload data from a pandas DataFrame to Google Cloud Storage as a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to upload.
    gcs : GCS
        The Google Cloud Storage client to use for the upload.
    dataset : str
        The name of the dataset in BigQuery.
    table_name : str
        The name of the table in BigQuery.
    updated_at : str
        The timestamp of the update.

    Returns
    -------
    blob: storage.Blob
        The Blob object representing the uploaded file.
    """
    blob = gcs.create_blob(f"{dataset}/{table_name}/{updated_at}.csv")
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
    return blob
