# The BareOps

## TODOs

- CI scripts not yet implemented.
- Refactor `pipeline_dev.py` to `pipeline.py` (`pipeline-training`) and update
    `Dockerfile`.

## DevOps

1. Setup DevOps local environment:
    1. Git
    2. Virtual Environment
    3. Setup files such as `pyproject.toml`, `setup.cfg`, `requirements.txt`,
       `Makefile`, `Dockerfile`, `docker-compose.yml`, `README.md`, etc. This
       can be set up using scripts.
    4. Styling/Formatter/Linter
    5. Logging with Rich (Monitoring locally). Important to have! See logger
       class.
    6. Pre-commit hooks
    7. Pytest
    8. Documentation
    9. Model serving
2. The steps in 1. should be reproduced via `Makefile`.
3. CI/CD: Currently just setup github actions to test the above workflows.

### Tests

Nest the unit, integration tests section under this section.

```bash
#!/bin/bash

#!/bin/bash

FOLDERS=(
    data_cleaning
    data_extraction
    data_loading
    data_resampling
    data_validation
    model_evaluation
    model_training
    model_validation_and_promotion
)

create_generic_tests_directories() {
    mkdir -p tests
    touch tests/conftest.py

    for dir in "${FOLDERS[@]}"
    do
        mkdir -p tests/unit/$dir
        touch tests/unit/$dir/__init__.py
        touch tests/unit/$dir/test_${dir}.py

        mkdir -p tests/integration/$dir
        touch tests/integration/$dir/__init__.py
        touch tests/integration/$dir/test_${dir}.py
    done

    # Create system tests directory and test_pipeline.py
    mkdir -p tests/system
    touch tests/system/test_pipeline.py
}

create_generic_tests_directories
```

### Code Formatting

For code formatting, you can use tools like
[black](https://black.readthedocs.io/). Black enforces a consistent code style
to make the codebase easier to read and understand.

### Linting

Linting tools like [pylint](https://www.pylint.org/) can be used to check your
code for potential errors and enforce a coding standard.

### Unit Testing

Unit testing frameworks like [pytest](https://docs.pytest.org/) can be used to
write tests that check the functionality of individual pieces of your code.

### Static Type Checking

Static type checkers like [mypy](http://mypy-lang.org/) can be used to perform
static type analysis on your code. This helps catch certain kinds of errors
before runtime.

### Integration Testing

Integration tests look at how different parts of your system work together.
These might be particularly important for ML workflows, where data pipelines,
training scripts, and evaluation scripts all need to interact smoothly.

### System Testing

System testing falls within the scope of black-box testing, and as such, should
require no knowledge of the inner design of the code or logic.

In a machine learning context, system testing might involve running the entire
machine learning pipeline with a predefined dataset and checking if the output
is as expected. You would typically look to see if the entire system, when run
end-to-end, produces the expected results, given a specific input. This could
involve evaluating overall system performance, checking the quality of the
predictions, and validating that the system meets all the specified
requirements.

## Performance Testing/Benchmarking

Track the performance of your models or certain parts of your code over time.
This could involve running certain benchmarks as part of your CI pipeline and
tracking the results.

### Model Validation

Depending on your workflow, you might want to have a stage that validates your
models, checking things like model performance metrics (accuracy, AUC-ROC, etc.)
to ensure they meet a certain threshold.

### Security Checks

Tools like [bandit](https://bandit.readthedocs.io/) can be used to find common
security issues in your Python code.

### Code Complexity Measurement

Tools like [radon](https://radon.readthedocs.io/) can give you metrics about how
complex your codebase is. This can help keep complexity down as the project
grows.

### Documentation Building and Testing

If you have auto-generated documentation, you might have a CI step to build and
test this documentation. Tools like [sphinx](https://www.sphinx-doc.org/) can
help with this.

### Scripts

Download all useful scripts.

```bash
curl -o scripts/clean.sh https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/clean.sh
```

### Makefile

Talks about `make clean` and `make create_repo_template` etc.

```bash
make venv
make clean
make create_repo_template
```

## DataOps

1. Extract -> Load -> Transform -> Load (ELTL) pure python.
2. Orchestrate with DAGs (self-implemented) and dockerize the pipeline.
3. CICD with GitHub Actions and deploy to Google Artifact Registry.
4. Deploy to Google Kubernetes Engine (GKE) and run the pipeline on `CronJob`.

## MLOps

### Local Development/Experimentation

Extract data from BigQuery to local environment. Note here the data is
transformed by dbt and pushed to BigQuery. This transform is not the same as the
transform in the preprocessing step of machine learning. This transform is just
to make the data agnostic to downstream tasks such as machine learning.

### Reproducibility

To ensure that your machine learning experiments are reproducible, you should
keep track of the following components:

1. **Code**
2. **Data**
3. **Model config, artifacts and metadata**

#### 1. Code versioning

Use a version control system like **Git** to keep track of your codebase. Git
allows you to track changes in your code over time and manage different
versions. To log the exact commit hash of your codebase when logging your MLflow
run, you can use the following code snippet:

```python
import subprocess

commit_hash = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
)
mlflow.log_param("commit_hash", commit_hash)
```

By logging the commit hash, you can always refer back to the exact version of
the code used for a specific run, ensuring reproducibility.

#### 2. Data versioning

For data versioning, you can use a tool like **DVC (Data Version Control)**. DVC
is designed to handle large data files, models, and metrics, and it integrates
well with Git. DVC helps you track changes in your data files and manage
different versions.

When you start a new MLflow run, log the DVC version or metadata of the input
data used in the experiment. This way, you can always retrieve the exact version
of the data used for a specific run, ensuring reproducibility.

See:

- [Data Management Tutorial](https://dvc.org/doc/start/data-management)

Important points:

- gitignore will be created automatically in data folder once you dvc add.
- After successfully pushing the data to remote, how do you "retrieve them"?
- If you are in the same repository, you can just pull the data from remote.

Yes, the idea is to use dvc checkout to switch between different versions of
your data files, as tracked by DVC. When you use dvc checkout, you provide a Git
commit hash or tag. DVC will then update your working directory with the data
files that were tracked at that specific Git commit.

Here are the steps to use dvc checkout with a Git commit hash:

Make sure you have the latest version of your repository and DVC remote by
running git pull and dvc pull.

Switch to the desired Git commit by running git checkout `<commit-hash>`.

Run dvc checkout to update your data files to the version tracked at the
specified commit.

Remember that dvc checkout only updates the data files tracked by DVC. To switch
between code versions, you'll still need to use git checkout.

```bash
git checkout <commit_hash>
dvc checkout # in this commit hash
dvc pull
```

#### 3. Model artifacts and metadata

You have already logged the artifacts (model, vectorizer, config, log files)
using `mlflow.log_artifact()`. You can also log additional metadata related to
the artifacts as you have done with additional_metadata. This should be
sufficient for keeping track of the artifacts associated with each run.

By combining code versioning with Git, data versioning with DVC, and logging
artifacts and metadata with MLflow, you can ensure that your machine learning
experiments are reproducible. This makes it easier to share your work,
collaborate with others, and build upon your experiments over time.

#### Recovering a run

1. Check the commit hashes for the code and data used in the run.
2. Checkout the code and data versions using the commit hashes.

```bash
git checkout <commit_hash>
pip install -r requirements.txt
python main.py train
# once done
git checkout main
```

## Experiment Tracking

### MLFlow Remote Tracking Server

Mirroring
[my MLFlow example's README](https://github.com/gao-hongnan/common-utils/tree/main/examples/containerization/docker/mlflow)
as well as my MLOps
[**documentation**](https://gao-hongnan.github.io/gaohn-mlops-docs/).

### Method 1. GCP VM

```bash
gcloud compute ssh --zone "asia-southeast1-a" "mlops-pipeline-v1" --project "gao-hongnan"
```

```bash
gaohn@<VM_NAME> $ git clone https://github.com/gao-hongnan/common-utils.git
```

```bash
gaohn@<VM_NAME> $ cd common-utils/examples/containerization/docker/mlflow
```

Then we echo something like the below to `.env` file.

```bash
echo -e "# Workspace storage for running jobs (logs, etc)\n\
WORKSPACE_ROOT=/tmp/workspace\n\
WORKSPACE_DOCKER_MOUNT=mlflow_workspace\n\
DB_DOCKER_MOUNT=mlflow_db\n\
\n\
# db\n\
POSTGRES_VERSION=13\n\
POSTGRES_DB=mlflow\n\
POSTGRES_USER=postgres\n\
POSTGRES_PASSWORD=mlflow\n\
POSTGRES_PORT=5432\n\
\n\
# mlflow\n\
MLFLOW_IMAGE=mlflow-docker-example\n\
MLFLOW_TAG=latest\n\
MLFLOW_PORT=5001\n\
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:password@db:5432/mlflow\n\
MLFLOW_ARTIFACT_STORE_URI=gs://gaohn/imdb/artifacts" > .env
```

Finally, run `bash build.sh` to build the docker image and run the container.

Once successful, you can then access the MLFlow UI at
`http://<EXTERNAL_VM_IP>:5001`.

#### The Model Registry

...

#### Promoter

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

## SSH

We create a new ssh key for this project to interact with GCP VM. We will make
it passwordless for convenience.

```bash
ssh-keygen -t rsa -f ~/.ssh/<SSH-FILE-NAME> -C "<USERNAME>"
```

A concrete implementation is as follows:

```bash
ssh-keygen -t rsa -f ~/.ssh/mlops-pipeline-v1 -C "gaohn"
```

And then in to ssh into the VM, we need to add the ssh key to the VM.

```bash
ssh <USERNAME>@<EXTERNAL-IP-ADDRESS>
```

and once in the VM, we need to add the ssh key to the VM.

```bash
cd ~/.ssh
```

Open the `authorized_keys` file and paste the public key in.

```bash
nano authorized_keys
```

Back on your local machine, open the `gcp_vm_no_passphrase.pub` file in a text
editor and copy its content.

Paste the content into the `authorized_keys` file on the VM.

Now one can ssh into the VM without password.

```bash
ssh -i ~/.ssh/<SSH-FILE-NAME> <USERNAME>@<EXTERNAL-IP-ADDRESS>
```

And to use the private ssh keys in github actions, we need to add the private
key to github secrets.

```bash
cat ~/.ssh/<SSH-FILE-NAME> | gh secret set <SSH-FILE-NAME> -b-
```

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

We will be using Google Kubernetes Engine (GKE) to deploy our application. As
usual, we assume that you have the google cloud sdk installed and configured.

On top of that, you need to install the
[`gke-gcloud-auth-plugin`](https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke):

```bash
gcloud components install gke-gcloud-auth-plugin
```

### Create a Kubernetes Cluster

As a first time user, I used the autopilot feature of Google Kubernetes Engine
(GKE) to create a cluster. The autopilot mode is a fully managed mode that
allows users to focus on deploying their applications without worrying about the
underlying infrastructure. It is a good choice for beginners.

Here is the command you provided broken down line by line for better
readability:

```bash
gcloud container --project "gao-hongnan" clusters create-auto "autopilot-cluster-2" --region "asia-southeast1" --release-channel "regular" --network "projects/gao-hongnan/global/networks/default" --subnetwork "projects/gao-hongnan/regions/asia-southeast1/subnetworks/default" --cluster-ipv4-cidr "/17" --services-ipv4-cidr "/22"
```

This command is creating an Autopilot cluster in GKE.

- `--project "gao-hongnan"` sets the GCP project to use.
- `clusters create-auto "autopilot-cluster-1"` creates an Autopilot cluster
    named `autopilot-cluster-1`.
- `--region "us-central1"` specifies the region where the cluster will be
    created.
- `--release-channel "regular"` specifies the release channel for the cluster.
- `--network` and `--subnetwork` are specifying the network and subnetwork for
    the cluster.
- `--cluster-ipv4-cidr "/17"` and `--services-ipv4-cidr "/22"` are specifying
    the IP address ranges for the cluster and the services within the cluster.

Remember to replace the project name and other parameters with your specific
values.

### Authenticate the Cluster

After creating the Kubernetes cluster using the `gcloud` command, the next step
would be to authenticate `kubectl` with the newly created cluster. You can do
this with the following command:

```bash
gcloud container clusters get-credentials autopilot-cluster-2 --region asia-southeast1 --project gao-hongnan
```

which outputs:

```markdown
Fetching cluster endpoint and auth data. kubeconfig entry generated for
autopilot-cluster-2.
```

This command fetches the access credentials for your cluster and automatically
configures `kubectl`. This allows you to use `kubectl` to interact with your
cluster.

### Verify the Cluster

```bash
kubectl get nodes
```

which outputs:

```markdown
❯ kubectl get nodes

NAME STATUS ROLES AGE VERSION gk3-autopilot-cluster-1-default-pool-0f7a9b55-wdgd
Ready <none> 10m v1.25.8-gke.1000
gk3-autopilot-cluster-1-default-pool-15d96b34-3szz Ready <none> 10m
v1.25.8-gke.1000
```

Why are there two nodes in the output?

In Kubernetes, a cluster consists of at least one master node and multiple
worker nodes. The master node (or control plane) manages the cluster, while the
worker nodes (or just nodes) are where your applications run.

When you create a cluster on Google Kubernetes Engine (GKE), GKE automatically
configures your cluster with multiple nodes for redundancy and high
availability. Each node is actually a separate virtual machine that is part of
your cluster.

So, even though you have one cluster, that cluster is made up of multiple nodes.
The command `kubectl get nodes` lists all the worker nodes in your cluster. In
your case, you have two worker nodes in your cluster, hence you see two nodes in
the output.

The fact that you have two nodes means that your cluster has more resources to
run your applications, and it also means that if one node fails, your
applications can still run on the other node.

### Create a ConfigMap

**Create a ConfigMap**: As explained in the previous posts, you need to create a
ConfigMap for your non-sensitive configuration data. Save the configuration data
in a `configmap.yaml` file and then use `kubectl apply -f configmap.yaml` to
create the ConfigMap in your Kubernetes cluster.

```bash
kubectl apply -f scripts/k8s/dataops/config_maps/configmap.yaml
```

which outputs:

```markdown
configmap/pipeline-dataops-config created
```

and to view it you can do:

```bash
kubectl get configmap pipeline-dataops-config
```

### Create a Secret

**Create a Secret**: Similarly, create a Secret for your sensitive configuration
data. Save the sensitive data in a `secret.yaml` file and then use
`kubectl apply -f secret.yaml` to create the Secret in your Kubernetes cluster.

```bash
kubectl create secret generic pipeline-dataops-secret \
--from-literal=GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$(cat /Users/gaohn/gaohn/TheBareOps/pipeline-dataops/gcp-storage-service-account.txt) \
--dry-run=client \
-o yaml > scripts/k8s/dataops/secrets/secret.yaml
```

Then apply it

```bash
kubectl apply -f scripts/k8s/dataops/secrets/secret.yaml
```

which returns

```markdown
secret/dataops-gcp-credentials-base64 created
```

### Mount the ConfigMap and Secret in a CronJob Manifest

**IMPORTANT** Below is an automated way to use CICD and deploy my
`manifests/cronjob.yaml` file. The `manifests/cronjob.yaml` file is a template
which requires you to change the GIT_COMMIT_HASH to the latest commit hash of
the repo. You can do this by running the following command:

```bash
sed -i 's|us-west2-docker.pkg.dev/gao-hongnan/thebareops/pipeline-dataops:.*|us-west2-docker.pkg.dev/gao-hongnan/thebareops/pipeline-dataops:'"$GIT_COMMIT_HASH"'|' test.yaml
```

```bash
kubectl apply -f -
```

**Create a CronJob**: Finally, you need to create a CronJob that uses the
ConfigMap and Secret you just created. Save the CronJob configuration in a
`cronjob.yaml` file and then use `kubectl apply -f cronjob.yaml` to create the
CronJob in your Kubernetes cluster.

```bash
kubectl apply -f scripts/k8s/dataops/manifests/dataops_cronjob.yaml
```

giving

```markdown
Warning: Autopilot set default resource requests for CronJob
default/dataops-cronjob, as resource requests were not specified. See
http://g.co/gke/autopilot-defaults cronjob.batch/dataops-cronjob created
```

This YAML file is a Kubernetes configuration that defines a `CronJob`. Let's
break down its structure:

- `apiVersion`: Specifies the version of the Kubernetes API that you're using
    to create this object.

- `kind`: Specifies the kind of resource you're creating. In this case, it's a
    `CronJob`, which is a job that runs on a regular schedule.

- `metadata`: Specifies metadata about the object, such as its name and
    labels.

- `spec`: Specifies the detailed configuration of the object. This can include
    a wide variety of settings, and it's where most of the configuration
    happens.

  - `schedule`: Specifies the schedule for the job in Cron format. In this
        case, it's set to run every 2 minutes.

  - `jobTemplate`: Specifies the template for the job, which includes the
        specification for the Pod that the job runs in.

    - `spec`: The specification for the Pod. This includes the containers
            that the Pod runs, their environment variables, and any volumes that
            they use.

      - `containers`: The list of containers to run in the Pod. Each
                container includes:

        - `name`: The name of the container.

        - `image`: The image to use for the container.

        - `imagePullPolicy`: The policy for pulling the image. This
                    can be `Always`, `Never`, or `IfNotPresent`. In this case,
                    it's set to `Always`, so the image is pulled every time the
                    Pod starts.

        - `env`: The environment variables for the container. Each
                    environment variable includes a name and a value, which can
                    be a literal value or a reference to a value stored in a
                    ConfigMap or a Secret.

        - `volumeMounts`: Specifies where to mount volumes in the
                    container's file system.

      - `volumes`: The volumes that the Pod uses. Each volume includes a
                name and a source, which can be a literal value or a reference
                to a value stored in a Secret, a ConfigMap, a
                PersistentVolumeClaim, or another type of volume source.

      - `restartPolicy`: Specifies the Pod's restart policy. In the
                event of a failure, this is set to `OnFailure`, so the Pod will
                be restarted.

If you have applied this configuration correctly, Kubernetes will create a new
`CronJob` resource according to these specifications.

After applying this configuration, you can monitor the status of the CronJob by
running `kubectl get cronjobs` and `kubectl get jobs`. This will list the
CronJobs and the Jobs they created.

If the CronJob is working correctly, it will create a new Job according to its
schedule (in this case, every 2 minutes). Each Job will create a Pod that runs
your specified container.

### Debugging

```bash
kubectl exec ...
```

### Watch the CronJob in Action

Great, it seems like your CronJob has been created successfully!

The warning message you see is because you have enabled Google Kubernetes Engine
(GKE) Autopilot, which automatically sets default resource requests for
workloads when resource requests are not specified.

Here are the steps for what you can do next:

0. `kubectl describe cronjob dataops-cronjob` to see the details of the CronJob
   or in general `kubectl describe pods` to check some errors especially if your
   CronJob is not running. Can watch your progress like

    ```markdown
    Events: Type Reason Age From Message ---- ------ ---- ---- ------- Normal
    Scheduled 11s gke.io/optimize-utilization-scheduler Successfully assigned
    default/dataops-cronjob-28130878-qtbxd to
    gk3-autopilot-cluster-1-pool-1-ffa4a760-q5lt Normal Pulling 2s kubelet
    Pulling image
    "us-west2-docker.pkg.dev/gao-hongnan/thebareops/pipeline-dataops:f7e1be0867bf7d8af67fd0f0da6b6fdb8b7e4346"
    ```

1. **Check the status of your CronJob**: You can use the `kubectl get cronjobs`
   command to list the CronJobs in the current namespace. The output should show
   the status of your CronJob including the last schedule time.

    ```bash
    kubectl get cronjobs
    ```

2. **View the Job created by the CronJob**: Once the CronJob has been triggered
   according to its schedule, it creates a Job. You can use `kubectl get jobs`
   to view the jobs that have been created. You can also use
   `kubectl describe job <job-name>` to view more details about a specific Job.

    ```bash
    kubectl get jobs
    kubectl describe job <job-name>
    ```

3. **View the logs of the Pods created by the Job**: Each Job creates one or
   more Pods to execute the task. You can use `kubectl get pods` to view the
   Pods and `kubectl logs <pod-name>` to view the logs of a specific Pod.

    ```bash
    kubectl get pods
    kubectl logs <pod-name>
    ```

    or if you want to monitor real time if the job is long.

    ```bash
    kubectl logs -f <pod-name>
    ```

4. **Check the result of your data pipeline**: Depending on what your data
   pipeline does, you might want to check the result. For example, if your data
   pipeline writes data into a database, you could check the contents of the
   database to ensure the data has been written correctly.

Remember that a CronJob will run according to the schedule you specified ("_/2
_ \* \* \*" in your case, which means every 2 minutes). Make sure your data
pipeline can complete within this interval to avoid overlapping runs.

You can use `kubectl logs <pod-name>` to check the logs of the Pods. Here you
can find the output of your script (`pipeline.py`), which will help you verify
whether the script executed successfully or not. If you see the expected output
in the logs, this indicates that the cronjob is successful.

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

To know more about what the command does, you can run the following commands to
see the documentation.

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

The folder structure of the mock dataops pipeline (without additional packages
like airbyte or dbt).
