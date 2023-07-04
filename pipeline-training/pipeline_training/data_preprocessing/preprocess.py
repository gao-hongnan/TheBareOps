from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC
from sklearn.preprocessing import StandardScaler

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.utils.common import get_file_format, get_file_size


class Preprocess:
    """
    Preprocess class for preprocessing the data.

    Parameters
    ----------
    cfg : Config
        Config object containing the configuration parameters.
    logger : Logger
        Logger object for logging information and errors.
    metadata : Metadata
        Metadata object containing the data to be preprocessed.
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

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be processed.

        Returns
        -------
        df : pd.DataFrame
            Processed DataFrame with handled missing values.
        """
        # implementation here
        return df

    def standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize numerical features in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be processed.

        Returns
        -------
        df : pd.DataFrame
            Processed DataFrame with standardized numerical features.
        """
        # implementation here
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be processed.

        Returns
        -------
        df : pd.DataFrame
            Processed DataFrame with encoded categorical features.
        """
        # implementation here
        return df

    def cast_columns(
        self, df: pd.DataFrame, column_types: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Cast features to appropriate data types.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be processed.

        Returns
        -------
        df : pd.DataFrame
            Processed DataFrame with casted features.
        """
        # implementation here
        # Cast the columns using the defined dictionary
        for col, new_type in column_types.items():
            df[col] = df[col].astype(new_type)

        return df

    def load_to_local(self, df: pd.DataFrame, output_filename: str) -> Path:
        filepath: Path = self.cfg.dirs.data.processed / f"{output_filename}.csv"
        df.to_csv(filepath, index=False)
        return filepath

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
        processed_dvc_metadata = self.dvc.add(filepath)
        try:
            self.dvc.push(filepath)
        except Exception as error:  # pylint: disable=broad-except
            self.logger.error(f"File is already tracked by DVC. Error: {error}")

        attr_dict = {"processed_dvc_metadata": processed_dvc_metadata}
        return attr_dict

    def run(self) -> Metadata:
        """
        Executes the data preprocessing process.

        Returns
        -------
        metadata : Metadata
            The updated Metadata object with information about the
            preprocessed data.
        """
        df = self.metadata.raw_df
        df = self.impute(df)
        df = self.cast_columns(
            df, **self.cfg.preprocess.cast_columns.model_dump(mode="python")
        )
        df = self.standardize_features(df)
        df = self.encode_categorical_features(df)

        # save processed data to local
        filepath = self.load_to_local(
            df, **self.cfg.preprocess.load_to_local.model_dump(mode="python")
        )
        attr_dict = {
            "processed_df": df,
            "processed_num_rows": df.shape[0],
            "processed_num_cols": df.shape[1],
            "processed_file_size": get_file_size(filepath=filepath),
            "processed_file_format": get_file_format(filepath=filepath),
        }

        if self.dvc is not None:
            track_with_dvc_metadata = self.track_with_dvc(Path(filepath))
            attr_dict.update(track_with_dvc_metadata)
        # update metadata
        self.metadata.set_attrs(attr_dict)
        return self.metadata
