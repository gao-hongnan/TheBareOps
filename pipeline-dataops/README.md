# DataOps

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

1. Encode the JSON file

```bash
❯ base64 -i ~/gcp-storage-service-account.json -o gcp-storage-service-account.txt
```

Copy this string to Github secrets or Kubernetes secrets depending on where you
are deploying.

2. Decode the JSON file

See entrypoint script.

### Build Docker Image Locally

You need to build using `linux/amd64` platform or else GKE will encounter an
[**error**](https://stackoverflow.com/questions/42494853/standard-init-linux-go178-exec-user-process-caused-exec-format-error).

```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
export IMAGE_NAME=pipeline-dataops && \
export IMAGE_TAG=$GIT_COMMIT_HASH && \
docker build \
--build-arg GIT_COMMIT_HASH=$GIT_COMMIT_HASH \
-f Dockerfile \
-t $IMAGE_NAME:$IMAGE_TAG \
.
```

Add `--platform linux/amd64` if you are building on a M1 Mac and want to push to
GCR.

### Run Docker Image Locally

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

### Pull and Test Run Locally

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

## DataOPs

- Extract data from source
- Load to staging GCS
  - The format is `dataset/table_name/created_at=YYYY-MM-DD:HH:MM:SS:MS` so
        that we can always find out which csv corresponds to which date in
        bigquery.
- Load to staging BigQuery
  - Write and Append mode! Incremental refresh
  - Added metadata such as `created_at` and `updated_at`
  - Bigquery has not so good primary key, ensure no duplicate in transforms
        step.
  - Add column coin type symbol in transform
  - TODO: use pydantic schema for data validation and creation.
- Transform data
- Load to production GCS
- Load to production BigQuery
- Query data from production BigQuery
  - This is the data that will be used for training and inference
  - Need to dvc here if possible
