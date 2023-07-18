# app/api.py
import os
import pickle
import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import mlflow
import pandas as pd
from common_utils.core.decorators.decorators import construct_response
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from rich.pretty import pprint

# from api.schemas.predictions import RatingInput, RatingOutput, Rating
# from api import schemas
# from api.config import get_settings

# TODO: DESERIALIZE CFG.PKL AND METADATA.PKL


class TradeData(BaseModel):
    open: float
    high: float
    low: float
    volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float


class TradeInput(BaseModel):
    data: List[TradeData]


class TradeOutput(BaseModel):
    prediction: List[int]
    probability: List[List[float]]


# # TODO: Check paul's usage of api_router decorator
# api_router = APIRouter()

# # disabling unused-argument and redefined-builtin because of FastAPI
# # pylint: disable=unused-argument,redefined-builtin
# # Define application
app = FastAPI(
    title="The Bare Ops",
    description="Predict price increase of a product",
    version="0.1",
)
tracking_uri = "http://mlflow:mlflow@34.142.130.3:5005/"
client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


@app.post("/predict", response_model=TradeOutput)
def predict(input: TradeInput) -> TradeOutput:
    # Convert the input data to a DataFrame
    df = pd.DataFrame([dict(item) for item in input.data])

    # Preprocess the input data
    df_preprocessed = preprocessor.transform(df)

    # Predict the output
    y_pred = model.predict(df_preprocessed)
    y_proba = model.predict_proba(df_preprocessed)

    # Return the prediction
    return TradeOutput(prediction=y_pred.tolist(), probability=y_proba.tolist())


# logged_model = 'runs:/a21e8ba1f674428188e654708113b090/registry'

# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)


# # Predict on a Pandas DataFrame.
# import pandas as pd
# loaded_model.predict(pd.DataFrame(data))
def get_latest_production_model(client, model_name):
    model = client.get_registered_model(model_name)
    for mv in model.latest_versions:
        if mv.current_stage == "Production":
            return mv
    return None


model_name = "thebareops_sgd"
latest_production_model = get_latest_production_model(client, model_name)
latest_production_version = latest_production_model.version
latest_production_run_id = latest_production_model.run_id
print(
    f"Latest production version of {model_name}: {latest_production_version} and run_id: {latest_production_run_id}"
)

latest_production_run = client.get_run(latest_production_run_id)
latest_production_experiment_id = latest_production_run.info.experiment_id
logged_model = f"gs://engr-nscc-mlflow-bucket/artifacts/{latest_production_experiment_id}/{latest_production_run_id}/artifacts/registry"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "~/engr-nscc-202304-ca07e4727134.json"
# model = mlflow.pyfunc.load_model(logged_model)
# print(model)
# TODO: super bad if i load from here i cannot use predict proba so i use below
model = mlflow.sklearn.load_model(logged_model)

artifacts = client.list_artifacts(latest_production_run_id)
pprint(artifacts)


stores_path = "stores"

stores_local_path = mlflow.artifacts.download_artifacts(
    run_id=latest_production_run_id,
    artifact_path=stores_path,
    tracking_uri=tracking_uri,
)
pprint(stores_local_path)
# FIXME: https://stackoverflow.com/questions/71059540/python-pickle-loadsdata-leads-to-modulenotfounderror-no-module-named-client
metadata = pickle.load(open(stores_local_path + "/artifacts/metadata.pkl", "rb"))
pprint(metadata)

artifacts = metadata.model_artifacts

X_test_original, y_test_original = metadata.X_test_original, metadata.y_test_original
pprint(X_test_original)
preprocessor = metadata.preprocessor
X_test = preprocessor.transform(X_test_original)
# label_encoder = metadata.label_encoder
from sklearn.preprocessing import LabelEncoder

y_test = LabelEncoder().fit_transform(y_test_original)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
pprint(y_pred)
pprint(y_proba)
from sklearn.metrics import accuracy_score, brier_score_loss

pprint(accuracy_score(y_test, y_pred))
pprint(brier_score_loss(y_test, y_proba[:, 1]))
