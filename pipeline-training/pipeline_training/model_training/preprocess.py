"""
Module to handle data preprocessing operations in a machine learning pipeline.

The Preprocessor class in this module is responsible for creating and configuring
various transformers needed for preprocessing the data such as imputers, scalers,
and encoders. These transformers are further used to create processing pipelines
for numeric and categorical data. The module is a key component of the pipeline
as it readies the raw data for model training.

Note: All transformers are created based on configurations defined in a Config object.

Attributes:
    Config (dataclass): Configuration dataclass object.
    Logger (class): Custom logging class.
    Metadata (class): Dataclass object to track inner pipeline state.
"""

from common_utils.core.logger import Logger
from sklearn import compose, impute, pipeline, preprocessing
from sklearn.impute._base import _BaseImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing._encoders import _BaseEncoder

from conf.base import Config
from metadata.core import Metadata


class Preprocessor:
    """
    A class to handle preprocessing operations in a machine learning pipeline.

    The class creates and configures imputers, scalers, and encoders based on the
    configurations defined in a Config object. It uses these transformers to create
    pipelines for numeric and categorical data.

    Attributes
    ----------
    cfg : Config
        Configuration object.
    logger : Logger
        Custom logger object.
    metadata : Metadata
        Object to track the state of the pipeline.
    """

    def __init__(self, cfg: Config, logger: Logger, metadata: Metadata) -> None:
        """
        Initialize the Preprocessor with a Config object, logger, and Metadata object.

        Parameters
        ----------
        cfg : Config
            Configuration object.
        logger : Logger
            Custom logger object.
        metadata : Metadata
            Object to track the state of the pipeline.
        """
        self.cfg = cfg
        self.logger = logger
        self.metadata = metadata

    def create_imputer(self) -> _BaseImputer:
        """
        Creates an imputer based on the configuration settings.

        Returns
        -------
        _BaseImputer
            Imputer object.
        """
        imputer_class = getattr(impute, self.cfg.train.create_imputer.name)
        imputer_args = self.cfg.train.create_imputer.model_dump(mode="python")
        imputer_args.pop("name")
        imputer = imputer_class(**imputer_args)
        return imputer

    def create_standard_scaler(self) -> StandardScaler:
        """
        Creates a standard scaler based on the configuration settings.

        Returns
        -------
        StandardScaler
            StandardScaler object.
        """
        scaler_class = getattr(
            preprocessing, self.cfg.train.create_standard_scaler.name
        )
        scaler_args = self.cfg.train.create_standard_scaler.model_dump(mode="python")
        scaler_args.pop("name")
        scaler = scaler_class(**scaler_args)
        return scaler

    def create_encoder(self) -> _BaseEncoder:
        """
        Creates an encoder based on the configuration settings.

        Returns
        -------
        _BaseEncoder
            Encoder object.
        """
        encoder_class = getattr(preprocessing, self.cfg.train.create_encoder.name)
        encoder_args = self.cfg.train.create_encoder.model_dump(mode="python")
        encoder_args.pop("name")
        encoder = encoder_class(**encoder_args)
        return encoder

    def create_numeric_transformer(self) -> pipeline.Pipeline:
        """
        Creates a pipeline for numeric transformations.

        Returns
        -------
        pipeline.Pipeline
            Pipeline for numeric transformations.
        """
        numeric_transformer = pipeline.Pipeline(
            steps=[
                ("imputer", self.create_imputer()),
                ("scaler", self.create_standard_scaler()),
            ]
        )
        return numeric_transformer

    def create_categorical_transformer(self) -> pipeline.Pipeline:
        """
        Creates a pipeline for categorical transformations.

        Returns
        -------
        pipeline.Pipeline
            Pipeline for categorical transformations.
        """
        categorical_transformer = pipeline.Pipeline(
            steps=[("encoder", self.create_encoder())]
        )
        return categorical_transformer

    def create_preprocessor(self) -> pipeline.Pipeline:
        """
        Creates a ColumnTransformer for preprocessing numeric and categorical features.

        The ColumnTransformer is updated in the Metadata object.

        Returns
        -------
        pipeline.Pipeline
            ColumnTransformer for preprocessing the data.
        """
        self.logger.info("Creating preprocessor...")
        numeric_features = self.cfg.train.features_and_targets.continuous_features
        categorical_features = self.cfg.train.features_and_targets.categorical_features

        self.logger.info(f"numeric_features: {numeric_features}")
        self.logger.info(f"categorical_features: {categorical_features}")

        numeric_transformer = self.create_numeric_transformer()
        categorical_transformer = self.create_categorical_transformer()

        preprocessor = compose.ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # Update metadata
        self.metadata.set_attrs(
            {
                "preprocessor": preprocessor,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
            }
        )
        return preprocessor
