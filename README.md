# MLOps

## Promoter


Indeed, comparing the newly trained model with the currently deployed model (often referred to as the "champion" model) is a common practice in many production settings. This is typically done by evaluating both models on the same validation dataset and comparing their performance metrics.

The MlflowClient provides a method get_model_version that you can use to retrieve details about a specific version of a registered model, including its metrics. If you've logged your model's performance metrics during training, you can retrieve these metrics and compare them with the performance of your new model.

Here's a simple example of how you might compare the new model to the currently deployed model:

```
def get_production_model_performance(model_name):
    client = MlflowClient()
    production_model = client.get_latest_versions(model_name, stages=["Production"])[0]
    return production_model.run_data.metrics

def compare_models(production_metrics, new_model_metrics):
    # Compare the performance metrics of the two models
    # This is a simple comparison, you might want to implement more robust model comparison logic.
    return new_model_metrics["accuracy"] > production_metrics["accuracy"]

def promote_model_to_production(model_name, new_model_metrics, new_model_version):
    production_metrics = get_production_model_performance(model_name)

    if not compare_models(production_metrics, new_model_metrics):
        print(f"New model did not outperform the current production model.")
        return

    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=new_model_version,
            stage="Production",
        )
        print(f"Model {model_name} version {new_model_version} is now in production.")
    except Exception as e:
        print(f"Failed to transition the model {model_name} version {new_model_version} to production. Error: {e}")

# call the function to promote model to production
promote_model_to_production("imdb", new_model_metrics, model_version.version)
```

promote_model_to_production
|
|---> if no model in production
|       |
|       ----> transition_model_to_production (deploy new model)
|
|---> if model in production
        |
        ----> get_production_model_performance (get current model metrics)
        |
        ----> compare_models (compare new model and current model)
                |
                ----> if new model is better
                        |
                        ----> transition_model_to_production (deploy new model)
