"""This clean/preprocess step is mostly cleaning and adding important columns such
as targets derived from the features.

For the real preprocessing such as imputing, encoding, and standardizing,
we will do it in the model training step to prevent data leakage."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from common_utils.core.logger import Logger
from common_utils.versioning.dvc.core import SimpleDVC

from conf.base import Config
from metadata.core import Metadata
from pipeline_training.utils.common import get_file_format, get_file_size


class Clean:
    """
    Cleaner class for preprocessing the data.

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
        """Loads the processed DataFrame to the local filesystem."""
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
            self.logger.info("File added to DVC and pushed to remote.")
        except Exception as error:  # pylint: disable=broad-except
            self.logger.error(f"File is already tracked by DVC. Error: {error}")

        attr_dict = {"processed_dvc_metadata": processed_dvc_metadata}
        return attr_dict

    def extract_features_and_target(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_columns: Union[str, List[str]],  # can be multi-label
        as_dataframe: bool = True,
    ) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """
        Returns the features and target from the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the features and target.

        Returns
        -------
        X : pd.DataFrame
            DataFrame containing the features.
        y : pd.DataFrame
            DataFrame containing the target.
        """
        X = df[feature_columns]
        y = df[target_columns]

        if not as_dataframe:
            X = X.values
            y = y.values
        attr_dict = {
            "X": X,
            "y": y,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
        }

        return attr_dict

    def feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering on the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to be processed.

        Returns
        -------
        df : pd.DataFrame
            Processed DataFrame with engineered features.
        """
        # implementation here
        df["price_increase"] = (df["close"] > df["open"]).astype(int)
        return df

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
        df = self.cast_columns(
            df, **self.cfg.clean.cast_columns.model_dump(mode="python")
        )
        df = self.feature_engineer(df)

        # save processed data to local
        filepath = self.load_to_local(
            df, **self.cfg.clean.load_to_local.model_dump(mode="python")
        )
        attr_dict = {
            "processed_df": df,
            "processed_num_rows": df.shape[0],
            "processed_num_cols": df.shape[1],
            "processed_file_size": get_file_size(filepath=filepath),
            "processed_file_format": get_file_format(filepath=filepath),
        }

        Xy = self.extract_features_and_target(
            df,
            **self.cfg.clean.extract_features_and_target.model_dump(mode="python"),
        )
        attr_dict.update(Xy)

        if self.dvc is not None:
            track_with_dvc_metadata = self.track_with_dvc(Path(filepath))
            attr_dict.update(track_with_dvc_metadata)
        # update metadata
        self.metadata.set_attrs(attr_dict)

        # release memory
        self.logger.info(
            "Preprocessing completed. Now releasing memory such as `raw_df`."
        )
        self.metadata.release("raw_df")
        assert self.metadata.raw_df is None
        return self.metadata
