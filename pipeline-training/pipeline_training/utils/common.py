from pathlib import Path


def get_file_size(filepath: Path) -> int:
    """Returns the size of a file in bytes."""
    return filepath.stat().st_size


def get_file_format(filepath: Path) -> str:
    """Returns the file format of a file."""
    return filepath.suffix[1:]
