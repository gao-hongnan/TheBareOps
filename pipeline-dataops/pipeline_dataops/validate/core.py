from typing import Any, Dict, List

import pandas as pd
import pydantic
from common_utils.core.logger import Logger

from schema.base import BaseSchema


def validate_schema(logger: Logger, df: pd.DataFrame, validator: BaseSchema) -> None:
    """
    Validate schema of a DataFrame using a Pydantic validator.

    This function converts a DataFrame to a list of dictionaries and
    applies the provided Pydantic validator to each dictionary. If any
    validation errors occur, they are logged and the error is raised.

    Parameters
    ----------
    logger : Logger
        An instance of Logger for logging.
    df : pd.DataFrame
        The DataFrame whose schema needs to be validated.
    validator : BaseSchema
        An instance of a Pydantic BaseSchema (or its subclass) that
        will be used for data validation.

    Raises
    ------
    pydantic.ValidationError
        If schema validation fails.
    """
    data_dict_list: List[Dict[str, Any]] = df.to_dict(
        "records"
    )  # Convert DataFrame to list of dictionaries
    try:
        _ = [validator.model_validate(data) for data in data_dict_list]
    except pydantic.ValidationError as error:
        logger.error(f"Schema validation failed with error: {error}")
        raise error


# TODO: technically pydantic already does this, but here we illustrate how to do it manually.
def validate_non_null_columns(
    logger: Logger, df: pd.DataFrame, columns: List[str]
) -> None:
    """
    Validate that a specific column in a DataFrame does not contain any null values.

    This function checks if the provided column in the DataFrame contains
    any null values. If any null values are found, it raises a ValueError.

    Parameters
    ----------
    logger : Logger
        An instance of Logger for logging.
    df : pd.DataFrame
        The DataFrame whose column needs to be checked for null values.
    column : str
        The column to check for null values.

    Raises
    ------
    ValueError
        If the column contains any null values.
    """
    for column in columns:
        if df[column].isnull().any():
            logger.error(f"Column {column} contains null values.")
            raise ValueError(f"Column {column} contains null values.")


# TODO: Below we validate data drifts / skews
# # Here, baseline_stats would be a dictionary mapping column names to their expected mean values
# # It would have been calculated during model training
# baseline_stats: Dict[str, float] = {}


# def validate_statistics(data: pd.DataFrame):
#     for column in data.columns:
#         if column in baseline_stats:
#             z_score = np.abs(
#                 (data[column].mean() - baseline_stats[column]) / data[column].std()
#             )
#             if (
#                 z_score > 3
#             ):  # Here 3 is just a commonly used threshold, adjust according to your specific needs
#                 print(
#                     f"Significant change detected in feature {column}. Consider retraining the model."
#                 )

# def expect_column_mean_to_be_within_range(
#     data: pd.DataFrame,
#     column: str,
#     min_val: Union[float, int],
#     max_val: Union[float, int],
# ):
#     mean_val = data[column].mean()
#     if not min_val <= mean_val <= max_val:
#         raise ValueError(f"Mean of column {column} is not within the expected range")
