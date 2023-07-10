# impute
# standardize_features
# encode_categorical_features
from sklearn.preprocessing import StandardScaler

from common_utils.core.logger import Logger
from conf.base import Config
from metadata.core import Metadata

from sklearn import compose, impute, pipeline, preprocessing
from sklearn.impute._base import _BaseImputer
from sklearn.impute import KNNImputer

from sklearn.preprocessing._encoders import _BaseEncoder


class Preprocessor:
    def __init__(self, cfg: Config, logger: Logger, metadata: Metadata) -> None:
        self.cfg = cfg
        self.logger = logger
        self.metadata = metadata

    def create_imputer(self) -> _BaseImputer:
        imputer_class = getattr(impute, self.cfg.train.create_imputer.name)
        imputer_args = self.cfg.train.create_imputer.model_dump(mode="python")
        imputer_args.pop("name")
        imputer = imputer_class(**imputer_args)
        return imputer

    def create_standard_scaler(self) -> StandardScaler:
        scaler_class = getattr(
            preprocessing, self.cfg.train.create_standard_scaler.name
        )
        scaler_args = self.cfg.train.create_standard_scaler.model_dump(mode="python")
        scaler_args.pop("name")
        scaler = scaler_class(**scaler_args)
        return scaler

    def create_encoder(self) -> _BaseEncoder:
        encoder_class = getattr(preprocessing, self.cfg.train.create_encoder.name)
        encoder_args = self.cfg.train.create_encoder.model_dump(mode="python")
        encoder_args.pop("name")
        encoder = encoder_class(**encoder_args)
        return encoder

    def create_numeric_transformer(self) -> pipeline.Pipeline:
        numeric_transformer = pipeline.Pipeline(
            steps=[
                ("imputer", self.create_imputer()),
                ("scaler", self.create_standard_scaler()),
            ]
        )
        return numeric_transformer

    def create_categorical_transformer(self) -> pipeline.Pipeline:
        categorical_transformer = pipeline.Pipeline(
            steps=[("encoder", self.create_encoder())]
        )
        return categorical_transformer

    def create_preprocessor(self) -> pipeline.Pipeline:
        self.logger.info("Creating preprocessor...")
        numeric_features = self.cfg.train.features.continuous_features
        categorical_features = self.cfg.train.features.categorical_features

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
        return preprocessor
