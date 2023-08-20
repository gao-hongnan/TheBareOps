# DataOps

- [DataOps](#dataops)
  - [Pinned Commit](#pinned-commit)
  - [Workflow](#workflow)
    - [Current Implementation](#current-implementation)
    - [The Full Stack Implementation](#the-full-stack-implementation)
  - [Setup Project Structure](#setup-project-structure)
  - [Setup Virtual Environment](#setup-virtual-environment)
  - [Containerization](#containerization)
    - [Secrets](#secrets)
    - [Docker Build](#docker-build)
      - [Preparing for the Build](#preparing-for-the-build)
      - [Building the Image](#building-the-image)
      - [Platform-Specific Building for GKE](#platform-specific-building-for-gke)
    - [Docker Run](#docker-run)
      - [Entrypoint](#entrypoint)
      - [Run Docker Image Locally](#run-docker-image-locally)
    - [Push Docker Image to Artifacts Registry](#push-docker-image-to-artifacts-registry)
    - [Deploy Docker Image from Artifacts Registry to Google Kubernetes Engine](#deploy-docker-image-from-artifacts-registry-to-google-kubernetes-engine)
    - [Pull and Test Run Locally](#pull-and-test-run-locally)
  - [Kubernetes](#kubernetes)
  - [Continuous Integration and Continuous Delivery](#continuous-integration-and-continuous-delivery)

See my mlops docs for details, but here is concrete implementation
so we need this as well for case study.

See gif made by Deepak Bhardwaj for a rough workflow.

![dataops-deepak-bhardwaj](./assets/dataops-deepak-bhardwaj.gif)

## Pinned Commit

```bash
commit df62949
```

is the working commit for this project.

## Workflow

Here's the steps involved in our DataOps workflow:

See
[DataOps Workflow](https://gao-hongnan.github.io/gaohn-mlops-docs/machine_learning_system_design/aiops/lifecycle/03_data_pipeline_data_engineering_and_dataops/) for details.

### Current Implementation

As a proof of concept, we merged the staging and production layers into one.

### The Full Stack Implementation

We can replace many parts with tech stacks such as Airbyte, dbt and Airflow.
This will be inside the "TheFullStackOps" repo.

## Setup Project Structure

```tree
tree -I 'venv|pipeline_dataops.egg-info|.git|.pytest_cache|tmp.*|__pycache__|__init__.py' -L 3 -a
```

```bash
#!/bin/bash

create_files() {
    touch .dockerignore \
          .env \
          .gitignore \
          Dockerfile \
          Makefile \
          README.md \
          pipeline.py \
          pyproject.toml \
          requirements.txt \
          requirements_dev.txt
}

create_conf_directories() {
    mkdir conf
    touch conf/__init__.py conf/base.py

    for dir in directory environment extract general load logger transform
    do
        mkdir conf/$dir
        touch conf/$dir/__init__.py conf/$dir/base.py
    done
}

create_metadata_directory() {
    mkdir metadata
    touch metadata/__init__.py metadata/core.py
}

create_pipeline_dataops_directories() {
    mkdir pipeline_dataops
    touch pipeline_dataops/__init__.py

    for dir in extract load transform validate
    do
        mkdir pipeline_dataops/$dir
        touch pipeline_dataops/$dir/__init__.py pipeline_dataops/$dir/core.py
    done
}

create_schema_directory() {
    mkdir schema
    touch schema/__init__.py schema/base.py schema/core.py
}

create_scripts_directories() {
    mkdir -p scripts/docker
    touch scripts/docker/entrypoint.sh

    mkdir -p scripts/k8s/dataops/config_maps scripts/k8s/dataops/manifests
}

create_tests_directories() {
    mkdir tests
    touch tests/conftest.py

    for dir in integration system unit
    do
        mkdir tests/$dir
        if [ "$dir" != "system" ]; then
            for subdir in extract load transform
            do
                mkdir tests/$dir/$subdir
                touch tests/$dir/$subdir/test_${subdir}.py
            done
        else
            touch tests/$dir/test_pipeline.py
        fi
    done
}

main() {
    create_files
    create_conf_directories
    create_metadata_directory
    create_pipeline_dataops_directories
    create_schema_directory
    create_scripts_directories
    create_tests_directories
}

main
```

## Setup Virtual Environment

```bash
cd pipeline-dataops && \
curl -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh && \
bash make_venv.sh venv --pyproject --dev && \
rm make_venv.sh && \
source venv/bin/activate
```

## Containerization

### Secrets

Handling sensitive information, such as service account credentials, in a secure
manner is crucial for maintaining the integrity and security of our applications
and data. In this section, we outline a method for securely handling such
secrets using base64 encoding and a secure storage location.

```bash
base64 -i ~/gcp-storage-service-account.json -o gcp-storage-service-account.txt
```

Here's a step-by-step breakdown of what is happening:

1. The `base64` command is encoding the `gcp-storage-service-account.json` file
   in base64 format. This encoding helps to ensure that the file data remains
   intact without modification during transport.

2. The output is redirected to `gcp-storage-service-account.txt` using the `-o`
   option.

3. This base64 string is then manually copied and pasted into either Github
   secrets or Kubernetes secrets, depending on where you are deploying your
   application.

Once you've added the secret to Github or Kubernetes, you can use it in your
workflows or manifests, respectively. These secrets will be masked and only
exposed to the processes that require them.

When the secret is required by your application, you'll need to decode it back
into its original form. For example, in a bash script, you might use the
following command:

```bash
echo "$YOUR_SECRET" | base64 --decode > gcp-storage-service-account.json
```

In this command, `YOUR_SECRET` would be the environment variable where your
base64 encoded secret is stored. This command decodes the secret and writes it
to `gcp-storage-service-account.json`.

### Docker Build

We will detail how to build a Docker image locally.

The commands below should be run from `~/<USER>/TheBareOps/pipeline-dataops/`.

First, I will show you the command to build the Docker image:

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD)
export IMAGE_NAME=pipeline-dataops
export IMAGE_TAG=$GIT_COMMIT_HASH
docker build \
--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH \
-f Dockerfile \
-t $IMAGE_NAME:$IMAGE_TAG \
.
```

Then, we show how to run the Docker image:

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
export HOME_DIR=/pipeline-dataops && \
export IMAGE_NAME=pipeline-dataops && \
export IMAGE_TAG=$GIT_COMMIT_HASH && \
export GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$(base64 -i gcp-storage-service-account.json)
docker run -it \
  --rm \
  --env PROJECT_ID="gao-hongnan" \
  --env GOOGLE_APPLICATION_CREDENTIALS="${HOME_DIR}/gcp-storage-service-account.json" \
  --env GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64 \
  --env GCS_BUCKET_NAME="gaohn" \
  --env GCS_BUCKET_PROJECT_NAME="thebareops_production" \
  --env BIGQUERY_RAW_DATASET=thebareops_production \
  --env BIGQUERY_RAW_TABLE_NAME=raw_binance_btcusdt_spot \
  --env BIGQUERY_TRANSFORMED_DATASET=thebareops_production \
  --env BIGQUERY_TRANSFORMED_TABLE_NAME=processed_binance_btcusdt_spot \
  --name $IMAGE_NAME \
  $IMAGE_NAME:$IMAGE_TAG
```

#### Preparing for the Build

Before initiating the build process, it's important to set specific variables
which are used during the build process:

- `GIT_COMMIT_HASH`: This is the Git commit hash of the current HEAD. It
    uniquely identifies the state of the codebase at the time of the build.

- `IMAGE_NAME`: This is the name you want to give to the Docker image. Here,
    we're using the name "pipeline-dataops".

- `IMAGE_TAG`: This is the tag for the Docker image. In this case, we're using
    the Git commit hash as the tag.

We use the following commands to set these variables:

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD)
export IMAGE_NAME=pipeline-dataops
export IMAGE_TAG=$GIT_COMMIT_HASH
```

#### Building the Image

The `docker build` command is used to build a Docker image from a Dockerfile and
a "context". The context is the set of files located in the specified PATH or
URL. In this case, the context is the current directory (represented by ".").

Here's the command to build the image:

```bash
docker build \
--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH \
-f Dockerfile \
-t $IMAGE_NAME:$IMAGE_TAG \
.
```

This command instructs Docker to:

- Use the Dockerfile in the current directory (`-f Dockerfile`).
- Name and tag the resulting image according to the `IMAGE_NAME` and
    `IMAGE_TAG` variables (`-t $IMAGE_NAME:$IMAGE_TAG`).
- Set the build argument `GIT_COMMIT_HASH` to the value of the
    `GIT_COMMIT_HASH` environment variable
    (`--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH`).

#### Platform-Specific Building for GKE

If you're developing on an Apple M1 Mac and plan to deploy the Docker image to
Google Kubernetes Engine (GKE), it's crucial to specify the
`--platform linux/amd64` flag during the build process. This is because the M1
Mac uses an ARM64 architecture, which differs from the x86_64 (or `linux/amd64`)
architecture that GKE typically runs on.

When a Docker image is built without specifying the target platform, it defaults
to the architecture of the build machine - ARM64 in the case of the M1 Mac.
However, trying to run an ARM64-built Docker image on GKE, which uses the
`linux/amd64` architecture, may result in a
[**compatibility error**](https://stackoverflow.com/questions/42494853/standard-init-linux-go178-exec-user-process-caused-exec-format-error).

To ensure your Docker image built on an M1 Mac is compatible with GKE, include
the `--platform linux/amd64` flag in your `docker build` command as follows:

```bash
docker build \
--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH \
--platform linux/amd64 \
-f Dockerfile \
-t $IMAGE_NAME:$IMAGE_TAG \
.
```

This step ensures that the image built is explicitly compatible with the
`linux/amd64` platform, circumventing potential compatibility issues when
deploying to GKE.

However, we typically build the image via a CI/CD pipeline, for instance, in a
Github Actions workflow. In this case, the architecture of the runner provided
by Github Actions is `linux/amd64`, which is compatible with GKE. This means
that the `--platform` flag isn't necessary when building the Docker image within
this CI/CD context.

### Docker Run

#### Entrypoint

An entrypoint script, typically written in Bash or Shell, is used in Docker to
prepare an environment for running your application and then actually start your
application. The script will be executed when the Docker container is started,
and it's the last thing that happens before your application's process starts.

Entrypoint scripts are powerful tools because they allow you to perform
operations that cannot be done in the Dockerfile, such as using environment
variables that are only available at runtime, executing commands that depend on
the state of the running container, or making last minute changes based on the
user's input when starting the container.

Here's a breakdown of the script you provided:

1. `#!/bin/sh`: This line is known as a shebang, and it specifies which
   interpreter should be used to run the script. In this case, it specifies the
   Bourne shell.

2. The script then checks if the file specified by the
   `GOOGLE_APPLICATION_CREDENTIALS` environment variable contains valid JSON. If
   the file does not contain valid JSON, the script assumes that the
   `GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64` environment variable contains a
   base64-encoded representation of the file, so it decodes it and writes the
   contents to the file specified by `GOOGLE_APPLICATION_CREDENTIALS`.

3. Finally, the script executes your application by running
   `python pipeline_dev.py`.

It's crucial to note a few important points regarding this script:

- You need to ensure that the `GOOGLE_APPLICATION_CREDENTIALS` and
    `GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64` environment variables are set
    before running the Docker container. You can do this using the `-e` or
    `--env` options in the `docker run` command, or by including the variables
    in an environment file that you pass to Docker using the `--env-file`
    option. If these variables are not set or are set incorrectly, the script
    might fail or behave unexpectedly.

- You also need to ensure that the `GOOGLE_APPLICATION_CREDENTIALS`
    environment variable specifies a valid path in the Docker container's file
    system. This file does not need to exist when the Docker container is
    started (since the script will create it), but the directory part of the
    path must exist. You might need to create the directory in your Dockerfile
    using the `RUN mkdir -p <dir>` command.

    For instance, if you specify
    `--env GOOGLE_APPLICATION_CREDENTIALS="/pipeline-training/gcp-storage-service-account.json"`
    then you need to ensure that the `/pipeline-training` directory exists in
    the Docker container's file system.

    For me I specify `HOME_DIR` as `/pipeline-training` and subsequently define
    `GOOGLE_APPLICATION_CREDENTIALS` as
    `${HOME_DIR}/gcp-storage-service-account.json` which is a valid path in the
    Docker container's file system.

- Lastly, it's important to note that the `exec` command is used to start your
    application. This replaces the current process (i.e., the shell running the
    entrypoint script) with your application's process. This is usually a good
    thing because it allows your application to receive signals sent to the
    Docker container (such as SIGTERM when Docker wants to stop the container),
    but it also means that no commands in the script after the `exec` command
    will be executed. Therefore, any cleanup or finalization code that you want
    to run when the application exits must be handled by the application itself,
    not the entrypoint script.

#### Run Docker Image Locally

Assuming your `gcp-storage-service-account.json` file is in the current
directory, you can run the Docker image locally using the following command:

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
export HOME_DIR=/pipeline-dataops && \
export IMAGE_NAME=pipeline-dataops && \
export IMAGE_TAG=$GIT_COMMIT_HASH && \
export GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$(base64 -i gcp-storage-service-account.json)
docker run -it \
  --rm \
  --env PROJECT_ID="gao-hongnan" \
  --env GOOGLE_APPLICATION_CREDENTIALS="${HOME_DIR}/gcp-storage-service-account.json" \
  --env GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64 \
  --env GCS_BUCKET_NAME="gaohn" \
  --env GCS_BUCKET_PROJECT_NAME="thebareops_production" \
  --env BIGQUERY_RAW_DATASET=thebareops_production \
  --env BIGQUERY_RAW_TABLE_NAME=raw_binance_btcusdt_spot \
  --env BIGQUERY_TRANSFORMED_DATASET=thebareops_production \
  --env BIGQUERY_TRANSFORMED_TABLE_NAME=processed_binance_btcusdt_spot \
  --name $IMAGE_NAME \
  $IMAGE_NAME:$IMAGE_TAG
```

And remember to pass `--build-arg HOME_DIR=YOUR_HOME_DIR` to `docker build` if
you want to use a different home directory.

### Push Docker Image to Artifacts Registry

Check `gar_docker_setup` in my `common-utils` on how to set up a container
registry in GCP.

First build the image again since we need it to be in `linux/amd64` format.

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
export PROJECT_ID="gao-hongnan" && \
export REPO_NAME="XXX" && \
export APP_NAME="XXX" && \
export REGION="XXX" && \
docker build \
--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH \
-f pipeline-training/Dockerfile \
--platform=linux/amd64 \
-t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}:${GIT_COMMIT_HASH}" \
.
```

Then push the image to the container registry.

```bash
docker push "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}:${GIT_COMMIT_HASH}" && \
echo "Successfully pushed ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}:${GIT_COMMIT_HASH}"
```

### Deploy Docker Image from Artifacts Registry to Google Kubernetes Engine

Note you do not need to provide the `GIT_COMMIT_HASH` as the image is already
tagged with the `GIT_COMMIT_HASH`.

- Set up `Expose` to false since this is not a web app.

### Pull and Test Run Locally

After you push the image to the container registry, you can pull the image and
run it locally to test.

```bash
export GIT_COMMIT_HASH=$(git rev-parse HEAD) && \
docker pull \
    us-west2-docker.pkg.dev/gao-hongnan/thebareops/pipeline-dataops:$GIT_COMMIT_HASH
```

```bash
cd pipeline-dataops && \
export GIT_COMMIT_HASH=$(git rev-parse HEAD) && \
export HOME_DIR=/pipeline-dataops && \
export IMAGE_NAME=pipeline-dataops && \
export IMAGE_TAG=$GIT_COMMIT_HASH && \
export GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$(base64 -i gcp-storage-service-account.json)
docker run -it \
  --rm \
  --env PROJECT_ID="gao-hongnan" \
  --env GOOGLE_APPLICATION_CREDENTIALS="${HOME_DIR}/gcp-storage-service-account.json" \
  --env GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64 \
  --env GCS_BUCKET_NAME="gaohn" \
  --env GCS_BUCKET_PROJECT_NAME="thebareops_production" \
  --env BIGQUERY_RAW_DATASET=thebareops_production \
  --env BIGQUERY_RAW_TABLE_NAME=raw_binance_btcusdt_spot \
  --env BIGQUERY_TRANSFORMED_DATASET=thebareops_production \
  --env BIGQUERY_TRANSFORMED_TABLE_NAME=processed_binance_btcusdt_spot \
  --name $IMAGE_NAME \
  us-west2-docker.pkg.dev/gao-hongnan/thebareops/pipeline-dataops:$IMAGE_TAG
```

## Kubernetes

...

## Continuous Integration and Continuous Delivery

...
