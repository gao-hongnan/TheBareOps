import logging
import time

import joblib
import pandas as pd
from common_utils.cloud.gcp.database.bigquery import BigQuery
from common_utils.cloud.gcp.storage.gcs import GCS
from common_utils.core.base import Connection, Storage
from common_utils.core.common import seed_all
from common_utils.core.logger import Logger
from rich.pretty import pprint

from conf.base import RUN_ID, Config
from conf.directory.base import ROOT_DIR

if __name__ == "__main__":
    cfg = Config()
    # pprint(cfg)
    seed_all(cfg.general.seed)

    # metadata = Metadata()
    # pprint(metadata)

    logger = Logger(
        log_file="pipeline_training.log",
        log_root_dir=cfg.dirs.stores.logs,
        module_name=__name__,
        propagate=False,
        level=logging.DEBUG,
    ).logger

    # log run_id and root_dir
    logger.info(f"run_id: {RUN_ID}\nroot_dir: {ROOT_DIR}")

    bq = BigQuery(
        project_id=cfg.env.project_id,
        google_application_credentials=cfg.env.google_application_credentials,
        dataset=cfg.env.bigquery_raw_dataset,
        table_name=cfg.env.bigquery_raw_table_name,
    )

    query = """
    SELECT *
    FROM `gao-hongnan.thebareops_production.processed_binance_btcusdt_spot` t
    WHERE t.utc_datetime > DATETIME(TIMESTAMP "2023-06-09 00:00:00 UTC")
    ORDER BY t.utc_datetime DESC
    LIMIT 1000;
    """
    df = bq.query(query=query, as_dataframe=True)

    df["price_increase"] = (df["close"] > df["open"]).astype(int)

    pprint(df.head())

    features = [
        "open",
        "high",
        "low",
        "volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    X = df[features]
    y = df["price_increase"]
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pprint(X_train.head())
    pprint(y_train.head())

    pprint(X_train.shape)
    pprint(X_test.shape)
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

    # save model to outputs folder
    joblib.dump(model, f"{cfg.dirs.stores.registry}/model.joblib")
