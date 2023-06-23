# DataOps

## Setup Project Structure

```bash
#!/bin/bash

# Create the directories
mkdir -p pipeline_dataops
mkdir -p conf
mkdir -p metadata

# Create inside pipeline_dataops
mkdir -p pipeline_dataops/extract
touch pipeline_dataops/extract/__init__.py

# Create the __init__.py files
touch pipeline_dataops/__init__.py
touch conf/__init__.py
touch metadata/__init__.py

# Create inside conf
mkdir -p conf/environment
mkdir -p conf/extract
mkdir -p conf/load
mkdir -p conf/transform
mkdir -p conf/directory

# Create the __init__.py files
touch conf/environment/__init__.py conf/environment/base.py
touch conf/extract/__init__.py conf/extract/base.py
touch conf/load/__init__.py conf/load/base.py
touch conf/transform/__init__.py conf/transform/base.py

# Create the .gitignore file
touch .gitignore
echo "# Add files and directories to be ignored by git here" > .gitignore

# Create .env file
touch .env
echo "# Add environment variables here" > .env

# Create the pyproject.toml file
touch pyproject.toml
```

## Setup Virtual Environment

```bash
curl -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh && \
bash make_venv.sh venv --pyproject --dev && \
rm make_venv.sh && \
source venv/bin/activate
```

## Containerization




#### Secrets

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
docker run -it \
    --env PROJECT_ID="${PROJECT_ID}" \
    --env GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS}" \
    --env GCS_BUCKET_NAME="${GCS_BUCKET_NAME}" \
    --env GCS_BUCKET_PROJECT_NAME="${GCS_BUCKET_PROJECT_NAME}" \
    --env BIGQUERY_RAW_DATASET="${BIGQUERY_RAW_DATASET}" \
    --env BIGQUERY_RAW_TABLE_NAME="${BIGQUERY_RAW_TABLE_NAME}" \
    pipeline-training:${GIT_COMMIT_HASH}

export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
export HOME_DIR=/pipeline-dataops && \
export IMAGE_NAME=pipeline-dataops && \
export IMAGE_TAG=$GIT_COMMIT_HASH && \
export GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$(base64 -i gcp-storage-service-account.json)
docker run -it \
  --rm \
  --env PROJECT_ID="gao-hongnan" \
  --env GOOGLE_APPLICATION_CREDENTIALS="${HOME_DIR}/gcp-storage-service-account.json" \
  --env GOOGLE_APPLICATION_CREDENTIALS_JSON=$GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64 \
  --env GCS_BUCKET_NAME="gaohn" \
  --env GCS_BUCKET_PROJECT_NAME="" \
  --env BIGQUERY_RAW_DATASET=thebareops_production \
  --env BIGQUERY_RAW_TABLE_NAME=raw_binance_btcusdt_spot \
  --env BIGQUERY_TRANSFORMED_DATASET=thebareops_production \
  --env BIGQUERY_TRANSFORMED_TABLE=processed_binance_btcusdt_spot \
  --name $IMAGE_NAME \
  $IMAGE_NAME:$IMAGE_TAG


docker exec -it <CONTAINER> /bin/bash
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
