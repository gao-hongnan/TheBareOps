# The BareOps

## Set GitHub Secrets and Variables from the Command Line

This guide provides instructions on how to set GitHub secrets and GitHub Actions
environment variables from the command line.

**GitHub secrets** are encrypted environment variables securely stored in your
repository settings. They're not shown in logs or to public forks and used for
storing sensitive information such as API keys or credentials.

**GitHub Actions environment variables** are stored in plaintext and are visible
in logs. These are used for storing non-sensitive information, like system paths
or feature flags.

### Setting GitHub Secrets

To set secrets in bulk, use a file with each line corresponding to a secret in
the format `SECRET_NAME=SECRET_VALUE`. This file can be named as you see fit,
but for this example, we'll call it `.env.github`.

Execute the following command:

```bash
ENV_FILE=.env.github
gh secret set -f $ENV_FILE
```

This command will read each line from `ENV_FILE` and set them as separate
secrets in your repository.

To set an individual secret, such as the contents of a JSON file, use the `-b`
flag to provide the secret's body. The content of the JSON file needs to be
passed as a string to the command. Be careful as the terminal treats the JSON
content as a string and may cause errors if not properly formatted.

```bash
JSON_CONTENT=$(cat ~/path_to_your_file/gcp-storage-service-account.json)
SECRET_NAME=GOOGLE_SERVICE_ACCOUNT_KEY
gh secret set $SECRET_NAME -b '$JSON_CONTENT'
```

### Setting GitHub Actions Environment Variables

In a similar fashion, you can set GitHub Actions environment variables in bulk
using a file with each line corresponding to a variable in the format
`VARIABLE_NAME=VARIABLE_VALUE`. This file can be named as you see fit, but for
this example, we'll call it `.env.github.variables`.

Use the following command:

```bash
ENV_FILE=.env.github.variables
gh variable set -f $ENV_FILE
```

This command will read the `ENV_FILE` and set each line in the file as a
separate GitHub Actions environment variable in your repository.

This way, you can manage your GitHub secrets and environment variables directly
from your terminal, improving your workflow's efficiency.

## Containerization

We will leverage Docker to containerize our application. For a comprehensive
understanding of Docker, please refer to this
[**detailed guide**](https://gao-hongnan.github.io/gaohn-mlops-docs/machine_learning_system_design/modern_tech_stacks/containerization/docker/concept/).

Just as producing high-quality code is essential, crafting well-structured
Dockerfiles is equally crucial. You can find a dedicated section on Dockerfile
best practices
[**here**](https://gao-hongnan.github.io/gaohn-mlops-docs/machine_learning_system_design/modern_tech_stacks/containerization/docker/concept/#docker-best-practices).

### Common Best Practices

1. Use a `.dockerignore` file to exclude files and directories from the build
   context.
2. Use multi-stage builds to keep the final image small and secure.
3. Use git commit hashes as tags for your images.

## Kubernetes

### Set Kubernetes Secrets from the Command Line

This guide provides instructions on how to set Kubernetes secrets from the
command line.

**Kubernetes Secrets** are objects that let you store and manage sensitive
information, such as passwords, OAuth tokens, and ssh keys. Storing confidential
information in a Secret is safer and more flexible than putting it verbatim in a
Pod definition or in a container image.

#### Setting Kubernetes Secrets

To set secrets in bulk, use an environment file with each line corresponding to
a secret in the format `SECRET_NAME=SECRET_VALUE`. This file can be named as you
see fit, but for this example, we'll call it `.env.k8s`.

Execute the following command:

```bash
ENV_FILE=.env.k8s
NAME=pipeline-dataops-secret # Name of the secret
kubectl create secret generic $NAME --from-env-file=$ENV_FILE
```

This command will read each line from the environment file and set them as
separate secrets in your Kubernetes namespace.

To know more about what the command does, you can run
the following commands to see the documentation.

```bash
kubectl create secret --help
kubectl create secret generic --help
```

If you want to set an individual secret, use the following command:

```bash
kubectl create secret generic my-secret --from-literal=MY_VARIABLE="Hello, world!"
```

This command will create a Secret named my-secret with a single entry named
MY_VARIABLE, whose value is "Hello, world!".

This way, you can manage your Kubernetes secrets directly from your terminal,
which can help you improve the efficiency of your Kubernetes workflows.

## MLOps

### Promoter

Indeed, comparing the newly trained model with the currently deployed model
(often referred to as the "champion" model) is a common practice in many
production settings. This is typically done by evaluating both models on the
same validation dataset and comparing their performance metrics.

The MlflowClient provides a method get_model_version that you can use to
retrieve details about a specific version of a registered model, including its
metrics. If you've logged your model's performance metrics during training, you
can retrieve these metrics and compare them with the performance of your new
model.

Here's a simple example of how you might compare the new model to the currently
deployed model:

```python
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

```tree
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
```
