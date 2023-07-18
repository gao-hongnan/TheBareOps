"""
Simple train-val-test split instead of K-Folds and its variants.
As mentioned in GCP blog, the output of data prepartion is
the data *splits* in the prepared format.

Generate balanced data splits.

NOTE: As emphasized this is not a modelling project. This is a project
to illustrate end-to-end MLOps. Therefore, we will not be using
a lot of "SOTA" methods.

Args:
    X (pd.Series): input features.
    y (np.ndarray): encoded labels.
    train_size (float, optional): proportion of data to use for training.
        Defaults to 0.7.
Returns:
    Tuple: data splits as Numpy arrays.
        X_train, X_val, X_test, y_train, y_val, y_test
# 70-15-15 split
"""
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import sklearn
from common_utils.core.logger import Logger

from conf.base import Config
from metadata.core import Metadata

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
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize Resampler with configuration, metadata, features, and targets.

        Parameters
        ----------
        cfg : Config
            Configuration object.
        metadata : Metadata
            Metadata object.
        X : np.ndarray, optional
            Features.
        y : np.ndarray, optional
            Targets.
        """
        self.cfg: Config = cfg
        self.metadata: Metadata = metadata
        self.logger: Logger = logger

        self.strategy_func = getattr(
            sklearn.model_selection, self.cfg.resample.strategy_name
        )
        self.X: Optional[np.ndarray] = X
        self.y: Optional[np.ndarray] = y
        if X is not None and y is not None:
            self.logger.info("Data provided in constructor.")
            self.prepare_data(X, y)

    def __getitem__(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specified subset.

        Parameters
        ----------
        key: str
            Subset key. Expected values are 'train', 'val', or 'test'.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Features and targets for the specified subset.
        """
        return self.dataset[key]

    def __len__(self) -> int:
        """
        Get the number of subsets.

        Returns
        -------
        int
            Number of subsets.
        """
        return len(self.dataset)

    # TODO: this should be abstractmethod since some users need to use their own
    # resampling strategy like StratifiedGroupKFold etc.
    def prepare_data(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> None:
        """
        Prepare data by splitting it into training, validation, and test sets.
        Naive train-val-test split, for sophisticated resampling override this method.

        Parameters
        ----------
        X : np.ndarray, optional
            New features to replace the instance variable, if any.
        y : np.ndarray, optional
            New targets to replace the instance variable, if any.
        """
        X = X if X is not None else self.X
        y = y if y is not None else self.y

        if X is None:
            raise ValueError("X is None.")

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
            "X_test_original": X_test,
            "y_test_original": y_test,
        }
        self.logger.info("Updating metadata.")
        self.metadata.set_attrs(attr_dict=attr_dict)

    def get_batches(
        self, key: str, batch_size: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate batches of data for a specified subset.

        Parameters
        ----------
        key : str
            Subset key. Expected values are 'train', 'val', or 'test'.
        batch_size : Optional[int]
            Size of batches. If None, entire dataset is used.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
            A batch of features and targets.
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
