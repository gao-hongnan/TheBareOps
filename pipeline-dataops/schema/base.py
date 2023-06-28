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
