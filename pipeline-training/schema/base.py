from __future__ import annotations

from typing import ClassVar, Dict, List, Type

from google.cloud import bigquery
from pydantic import BaseModel


class BaseSchema(BaseModel):
    pydantic_to_bq_types: ClassVar[Dict[str, bigquery.enums.SqlTypeNames]] = {
        "int": bigquery.enums.SqlTypeNames.INT64,
        "float": bigquery.enums.SqlTypeNames.FLOAT64,
        "str": bigquery.enums.SqlTypeNames.STRING,
        "bool": bigquery.enums.SqlTypeNames.BOOL,
        "datetime": bigquery.enums.SqlTypeNames.DATETIME,
    }

    pydantic_to_pd_types: ClassVar[Dict[str, str]] = {
        "int": "int64",
        "float": "float64",
        "str": "object",
        "bool": "bool",
        "datetime": "datetime64[ns]",
    }

    @classmethod
    def to_bq_schema(cls: Type[BaseSchema]) -> List[bigquery.SchemaField]:
        schema = []
        for name, field in cls.__annotations__.items():
            field_type = field.__name__
            if field_type not in cls.pydantic_to_bq_types:
                raise ValueError(
                    f"Cannot convert {field_type} to a BigQuery data type."
                )

            bq_field_type = cls.pydantic_to_bq_types[field_type]
            schema_field = bigquery.SchemaField(name, bq_field_type, mode="NULLABLE")
            schema.append(schema_field)

        return schema

    @classmethod
    def to_pd_dtypes(cls: Type[BaseSchema]) -> Dict[str, str]:
        dtypes = {}
        for name, field in cls.__annotations__.items():
            field_type = field.__name__
            if field_type not in cls.pydantic_to_pd_types:
                raise ValueError(f"Cannot convert {field_type} to a Pandas data type.")

            pd_dtype = cls.pydantic_to_pd_types[field_type]
            dtypes[name] = pd_dtype

        return dtypes
