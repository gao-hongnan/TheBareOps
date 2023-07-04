from pathlib import Path
from typing import Any, Dict, Optional, Union

from common_utils.core.decorators.timer import timer
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.utils.common import get_file_format, get_file_size


class Load:
    """
    Load class for loading data from metadata, saving it to a local file,
    and optionally tracking the data file using DVC.

    Parameters
    ----------
    cfg : Config
        Config object containing the configuration parameters.
    logger : Logger
        Logger object for logging information and errors.
    metadata : Metadata
        Metadata object containing the data to be loaded.
    dvc : SimpleDVC, optional
        DVC object for data file tracking (default is None).
    """

    def __init__(
        self,
        cfg: Config,
        logger: Logger,
        metadata: Metadata,
        dvc: Optional[SimpleDVC] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.metadata = metadata
        self.dvc = dvc

    def load_to_local(
        self, table_name: str, output_filename: str
    ) -> Dict[str, Union[int, str]]:
        """
        This function loads data from the metadata into a local CSV file, and
        optionally tracks that file using Data Version Control (DVC).

        Design Principles:
        ------------------
        1. Single Responsibility Principle (SRP): This function solely
        handles loading data and potentially tracking it with DVC.

        2. Dependency Inversion Principle (DIP): Dependencies (logger,
        directories, metadata, and optionally DVC) are passed as
        arguments for better testability and flexibility.

        3. Open-Closed Principle (OCP): The function supports optional DVC
        tracking, showing it can be extended without modifying the
        existing code. In other words, the function is open for
        extension because it can be extended to support DVC tracking,
        and closed for modification because the existing code does not
        need to be modified to support DVC tracking.

        4. Use of Type Hints: Type hints are used consistently for clarity
        and to catch type-related errors early.

        5. Logging: Effective use of logging provides transparency and aids
        in troubleshooting.


        Areas of Improvement:
        ---------------------
        1. Exception Handling: More specific exception handling could
        improve error management.

        2. Liskov Substitution Principle (LSP): Using interfaces or base
        classes for the inputs could enhance flexibility and adherence
        to LSP.

        3. Encapsulation: Consider if direct manipulation of `metadata`
        attributes should be encapsulated within `metadata` methods.
        """
        self.logger.info("Reading data from metadata computed in the previous step...")
        raw_df = self.metadata.raw_df

        assert (
            table_name == self.metadata.raw_table_name
        ), f"Table name mismatch: Expected {self.metadata.raw_table_name}, got {table_name}."

        filepath: Path = self.cfg.dirs.data.raw / f"{output_filename}.csv"

        raw_df.to_csv(filepath, index=False)

        attr_dict = {
            "raw_file_size": get_file_size(filepath),
            "raw_file_format": get_file_format(filepath),
            "raw_file_path": str(filepath),
        }
        return attr_dict

    def track_with_dvc(self, filepath: Path) -> Dict[str, Dict[str, Any]]:
        """
        Adds a local file to DVC and attempts to push it.

        Parameters
        ----------
        filepath : Path
            Path to the file to be added to DVC.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the metadata returned by DVC.
        """
        # add local file to dvc
        raw_dvc_metadata = self.dvc.add(filepath)
        try:
            self.dvc.push(filepath)
            self.logger.info("File added to DVC and pushed to remote.")
        except Exception as error:  # pylint: disable=broad-except
            self.logger.error(f"File is already tracked by DVC. Error: {error}")

        attr_dict = {"raw_dvc_metadata": raw_dvc_metadata}
        return attr_dict

    @timer(display_table=True)
    def run(self) -> Metadata:
        """
        Executes the data loading and optional DVC tracking process.

        Returns
        -------
        Metadata
            The updated Metadata object with information about the
            loaded and optionally DVC-tracked file.
        """
        load_to_local_metadata = self.load_to_local(
            **self.cfg.load.load_to_local.model_dump(mode="python")
        )

        if self.dvc is not None:
            track_with_dvc_metadata = self.track_with_dvc(
                Path(load_to_local_metadata["raw_file_path"])
            )
            load_to_local_metadata.update(track_with_dvc_metadata)
        self.metadata.set_attrs(load_to_local_metadata)
        return self.metadata
