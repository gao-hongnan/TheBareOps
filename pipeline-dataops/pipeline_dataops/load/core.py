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
    job_config = bq.load_job_config(schema=schema, write_disposition=write_disposition)
    bq.load_table_from_dataframe(df=df, job_config=job_config)


def to_google_cloud_storage(
    df: pd.DataFrame, gcs: GCS, dataset: str, table_name: str, updated_at: str
) -> storage.Blob:
    blob = gcs.create_blob(f"{dataset}/{table_name}/{updated_at}.csv")
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")
    return blob
