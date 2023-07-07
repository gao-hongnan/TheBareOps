"""
Simple train-val-test split instead of K-Folds and its variants.
As mentioned in GCP blog, the output of data prepartion is
the data *splits* in the prepared format.
"""
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.logger import Logger
from typing import Dict, Generator, Optional, Tuple
from sklearn import model_selection

import numpy as np
import sklearn

from conf.base import Config
from metadata.core import Metadata

# TODO: Consider adding a class and do generator style.


# TODO: this should be abstractmethod since some users need to use their own
# resampling strategy like StratifiedGroupKFold etc.

# NOTE: This is not similar to PyTorch Dataset or DataLoader.
# It is merely using getitem to get "splits" of data.
# The real generator is get_batches.


class Resampler:
    """
    Resampler class for generating balanced data splits.
    """

    def __init__(
        self,
        cfg: Config,
        metadata: Metadata,
        logger: Logger,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Initialize Resampler with configuration, metadata, features, and targets.

        Args:
            cfg (Config): Configuration object.
            metadata (Metadata): Metadata object.
            X (np.ndarray): Features.
            y (np.ndarray): Targets.
        """
        self.cfg: Config = cfg
        self.metadata: Metadata = metadata
        self.logger: Logger = logger

        self.strategy_func = getattr(
            sklearn.model_selection, self.cfg.resample.strategy_name
        )
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.prepare_data()
        self.logger.info("Data prepared in constructor.")

    def __getitem__(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specified subset.

        Parameters
        ----------
        key: str
            Subset key. Expected values are 'train', 'val', or 'test'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]: Features and targets for the specified subset.
        """
        return self.dataset[key]

    def __len__(self) -> int:
        """
        Get the number of subsets.

        Returns:
            int: Number of subsets.
        """
        return len(self.dataset)

    def prepare_data(self) -> None:
        """
        Prepare data by splitting it into training, validation, and test sets.
        This is a naive train-val-test split, for sophisticated resampling
        need to override this method.
        """
        X_train, X_, y_train, y_ = self.strategy_func(
            self.X,
            self.y,
            stratify=self.y,
            **self.cfg.resample.strategy_config.initial_split,
        )
        X_val, X_test, y_val, y_test = self.strategy_func(
            X_, y_, stratify=y_, **self.cfg.resample.strategy_config.secondary_split
        )
        self.dataset: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

        # Update metadata
        attr_dict = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
        self.logger.info("Updating metadata.")
        self.metadata.set_attrs(attr_dict=attr_dict)

    def get_batches(
        self, key: str, batch_size: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of data for a specified subset.

        Args:
            key (str): Subset key. Expected values are 'train', 'val', or 'test'.
            batch_size (Optional[int]): Size of batches. If None, entire dataset is used.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A batch of features and targets.
        """
        X, y = self.dataset[key]
        n_samples = X.shape[0]

        if batch_size is None:
            self.logger.warning(
                "`batch_size` is set to None. Using entire dataset as batch."
            )
            yield X, y
        else:
            # Shuffle data
            # You need to shuffle the indices still which is different from
            # shuffling the data.
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]
                yield X_batch, y_batch


def get_data_splits(
    cfg: Config, metadata: Metadata, X: np.ndarray, y: np.ndarray
) -> Metadata:
    """Generate balanced data splits.

    NOTE: As emphasized this is not a modelling project. This is a project
    to illustrate end-to-end MLOps. Therefore, we will not be using
    a lot of "SOTA" methods.

    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.
    Returns:
        Tuple: data splits as Numpy arrays.
    """
    # 70-15-15 split

    strategy_func = getattr(model_selection, cfg.resample.strategy_name)

    X_train, X_, y_train, y_ = strategy_func(
        X, y, stratify=y, **cfg.resample.strategy_config.initial_split
    )
    X_val, X_test, y_val, y_test = strategy_func(
        X_, y_, stratify=y_, **cfg.resample.strategy_config.secondary_split
    )

    attr_dict = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    metadata.set_attrs(attr_dict=attr_dict)
    return metadata
