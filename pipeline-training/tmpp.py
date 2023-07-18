import mlflow
import time

client = mlflow.tracking.MlflowClient(
    tracking_uri="http://mlflow:mlflow@34.142.130.3:5005/"
)
# exp_id = client.get_experiment_by_name("thebareops_sgd_study").experiment_id
# client.delete_experiment(exp_id)
# client.delete_experiment(exp_id)
# client.delete_experiment("3")
client.transition_model_version_stage(
    name="thebareops_sgd",
    version="3",
    stage="None",  # Or "Staging", "Archived"
)

time.sleep(100)
