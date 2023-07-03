from pydantic import BaseModel, Field
from rich.pretty import pprint


class ExtractFromDataWarehouse(BaseModel):
    """
    Pydantic model for SQL query used to extract data from the data warehouse.

    Attributes
    ----------
    query : str
        SQL query to fetch data from the database.
    """

    query: str = Field(
        default="""
    SELECT *
    FROM `gao-hongnan.thebareops_production.processed_binance_btcusdt_spot` t
    WHERE t.utc_datetime > DATETIME(TIMESTAMP "2023-06-09 00:00:00 UTC")
    ORDER BY t.utc_datetime DESC
    LIMIT 1000;
    """,
        description="SQL query to fetch data from the database.",
    )


class Extract(BaseModel):
    """
    Pydantic model for Data Extraction configurations. It is composed of all
    configurations belonging to functions or classes in data_extraction/extract.py.

    Attributes
    ----------
    extract_from_data_warehouse : ExtractFromDataWarehouse
        Instance of ExtractFromDataWarehouse containing the SQL query.
    """

    extract_from_data_warehouse: ExtractFromDataWarehouse = Field(
        default=ExtractFromDataWarehouse(),
        description="Instance of ExtractFromDataWarehouse.",
    )


if __name__ == "__main__":
    extract = Extract()
    pprint(extract)
