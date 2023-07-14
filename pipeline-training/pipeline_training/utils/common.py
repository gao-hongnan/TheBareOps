from pathlib import Path
from typing import Dict
import pandas as pd
from prettytable import PrettyTable


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
