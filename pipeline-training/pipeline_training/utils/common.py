from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from common_utils.core.logger import Logger
from prettytable import PrettyTable


def compare_test_case(
    actual: Any,
    expected: Any,
    description: str = "",
    logger: Optional[Logger] = None,
) -> None:
    try:
        if isinstance(actual, pd.DataFrame) or isinstance(actual, pd.Series):
            assert actual.equals(expected)
        elif isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected

        message = f"[green]Test passed:[/green] {description}"
        # If a logger is provided, log the message
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    except AssertionError:
        message = f"[red]Test failed:[/red] {description}\nExpected: {expected}, but got: {actual}"
        if logger is not None:
            logger.error(message)
        else:
            print(message)


def compare_test_cases(
    actual_list: List[Any],
    expected_list: List[Any],
    description_list: List[str],
    logger: Optional[Logger] = None,
) -> None:
    assert len(actual_list) == len(
        expected_list
    ), "Lengths of actual and expected are different."

    for i, (actual, expected, description) in enumerate(
        zip(actual_list, expected_list, description_list)
    ):
        compare_test_case(
            actual=actual,
            expected=expected,
            description=f"{description} - {i}",
            logger=logger,
        )


def get_file_size(filepath: Path) -> int:
    """Returns the size of a file in bytes."""
    return filepath.stat().st_size


def get_file_format(filepath: Path) -> str:
    """Returns the file format of a file."""
    return filepath.suffix[1:]


def log_data_splits_summary(
    splits: Dict[str, pd.DataFrame], total_size: int = None
) -> PrettyTable:
    # Create a pretty table
    table = PrettyTable()
    table.field_names = ["Data Split", "Size", "Percentage"]
    table.align = "l"

    for split_name, split_data in splits.items():
        percentage = (len(split_data) / total_size) * 100
        table.add_row([split_name, len(split_data), f"{percentage:.2f}%"])

    return table
