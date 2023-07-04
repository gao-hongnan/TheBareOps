from pydantic import BaseModel, Field
from rich.pretty import pprint


class LoadToLocal(BaseModel):
    """
    Pydantic model for SQL query used to extract data from the data warehouse
    and load it to the local machine.

    Attributes
    ----------
    query : str
        SQL query to fetch data from the database.
    """

    table_name: str = Field(
        default="raw_binance_btcusdt_spot",
        description="The table name to load data from.",
    )
    output_filename: str = Field(
        default="raw_binance_btcusdt_spot", description="Output filename."
    )


class Load(BaseModel):
    """
    Pydantic model for Data Extraction configurations. It is composed of all
    configurations belonging to functions or classes in data_extraction/extract.py.

    Attributes
    ----------
    extract_from_data_warehouse : ExtractFromDataWarehouse
        Instance of ExtractFromDataWarehouse containing the SQL query.
    """

    load_to_local: LoadToLocal = Field(
        default=LoadToLocal(), description="Instance of LoadToLocal."
    )
