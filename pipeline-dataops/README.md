# DataOps

Here's the steps involved in our DataOps workflow:

1. **Data Extraction**

    - Source data is identified and extracted from various internal and external
      databases and APIs.

2. **Loading Data to Staging Google Cloud Storage (GCS)**

    - Extracted data is loaded into a dedicated staging area within GCS. This
      process serves as the initial raw data checkpoint, providing an immutable
      storage layer for unprocessed data. This approach to storing raw data
      helps maintain data integrity throughout the pipeline.
    - The data is stored in a structured format, for instance, in the form of:

        ```text
        dataset/table_name/created_at=YYYY-MM-DD:HH:MM:SS:MS`
        ```

        This structure allows for easy tracking of the data's origin and
        timestamp.

3. **Loading Data to Staging BigQuery**

    - The data in the staging GCS is loaded into Google BigQuery for more
      advanced processing and analysis.
    - Data is loaded using both write and append modes, allowing for incremental
      refreshes.
    - Metadata such as `created_at` and `updated_at` timestamps are added to
      maintain a detailed record of the data's lineage.
    - As BigQuery's primary key system may have limitations, one needs to be
      careful to ensure that there are no **duplicate** records in the data.

4. **Data Validation After Extraction**

    - Once the data is extracted and loaded into the staging area in GCS or
      BigQuery, a preliminary data validation process is conducted.
    - This may include checking for the presence and correctness of key fields,
      ensuring the right data types, checking data ranges, verifying data
      integrity, and so on.
    - If the data fails the validation, appropriate error handling procedures
      should be implemented. This may include logging the error, sending an
      alert, or even stopping the pipeline based on the severity of the issue.

5. **Data Transformation**

    - In this step, the raw data from the staging area undergoes a series of
      transformation processes to be refined into a format suitable for
      downstream use cases, including analysis and machine learning model
      training. These transformations might involve operations such as:

        - **Data Cleaning**: Identifying and correcting (or removing) errors and
          inconsistencies in the data. This might include handling missing
          values, eliminating duplicates, and dealing with outliers.

        - **Joining Data**: Combining related data from different sources or
          tables to create a cohesive, unified dataset.

        - **Aggregating Data**: Grouping data by certain variables and
          calculating aggregate measures (such as sums, averages, maximum or
          minimum values) over each group.

        - **Structuring Data**: Formatting and organizing the data in a way
          that's appropriate for the intended use cases. This might involve
          creating certain derived variables, transforming data types, or
          reshaping the data structure.

    - It's important to note that the transformed data at this stage is intended
      to be a high-quality, flexible data resource that can be leveraged across
      a range of downstream use cases - not just for machine learning model
      training and inference. For example, it might also be used for business
      reporting, exploratory data analysis, or statistical studies.

    - By maintaining a general-purpose transformed data layer, the pipeline
      ensures that a broad array of users and applications can benefit from the
      data cleaning and transformation efforts, enhancing overall data usability
      and efficiency within the organization.

6. **Load Transformed Data to Staging GCS and BigQuery**

    - After the data transformation and validation, the resulting data is loaded
      back into the staging environment. This involves both Google Cloud Storage
      (GCS) and BigQuery.

        - **Staging GCS**: The transformed data is saved back into a specific
          location in the staging GCS. This provides a backup of the transformed
          data and serves as an intermediate checkpoint before moving the data
          to the production layer.

        - **Staging BigQuery**: The transformed data is also loaded into a
          specific table in the staging area in BigQuery. Loading the
          transformed data into BigQuery allows for quick and easy analysis and
          validation of the transformed data, thanks to BigQuery's capabilities
          for handling large-scale data and performing fast SQL-like queries.

    - This step of loading the transformed data back into the staging GCS and
      BigQuery is crucial for maintaining a robust and reliable data pipeline.
      It ensures that the transformed data is correctly saved and accessible for
      future steps, and it allows for any necessary checks or reviews to be
      performed before the data is moved to the production environment.

7. **Data Validation After Transformation**

    - After the data transformation process, another round of validation is
      carried out on the transformed data.
    - This may involve checking the output of the transformation against
      expected results, ensuring the data structure conforms to the target
      schema, and performing statistical checks (e.g., distributions,
      correlations, etc.).
    - If the transformed data fails the validation, appropriate steps are taken
      just like after extraction.

Next in the pipeline, once the transformed and validated data in the staging
layer has been confirmed to be accurate and ready for use, it would then be
moved to the production layer. This would be handled in the following steps:

1. **Loading Data to Production GCS**

    - The validated and transformed data is moved from the staging GCS to the
      production GCS. This signals that the data is ready for use in downstream
      applications and processes.

2. **Loading Data to Production BigQuery**

    - Similarly, the validated and transformed data in BigQuery is moved from
      the staging dataset to the production dataset. This makes the data
      available for querying, reporting, machine learning model training, and
      other use cases.

3. **Querying Data from Production BigQuery**

    - Finally, data is queried from the production dataset in BigQuery for use
      in various downstream applications, such as training and inference for
      machine learning models, data analysis, reporting, and more.

Typically, the movement of data from the staging layer to the production layer
happens once the data has been cleaned, transformed, validated, and is deemed
ready for use in downstream applications such as machine learning model
training, analytics, reporting, etc. The transformed data is first validated to
ensure that it meets the required quality standards. If the validation is
successful, the data is moved to the production layer. The goal is to only
expose clean, validated, and reliable data to end users or downstream
applications.

Once the data has passed both rounds of validation, it can be loaded into the
production layer in both GCS and BigQuery. At this point, the data is ready for
downstream use in tasks such as model training and inference.

In the context of ML, these steps form the beginning part of our pipeline, where
data is extracted, cleaned, and made ready for use in our ML models. Each step
is designed to ensure the integrity and usability of the data, from extraction
to querying for model training and inference.

## Setup Project Structure

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

## Continuous Integration and Continuous Delivery

...