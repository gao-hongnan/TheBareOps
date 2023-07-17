
```bash
export GIT_COMMIT_HASH=$(git rev-parse --short HEAD) && \
export HOME_DIR=/pipeline-training && \
export IMAGE_NAME=pipeline-training && \
export IMAGE_TAG=$GIT_COMMIT_HASH && \
export GOOGLE_APPLICATION_CREDENTIALS_JSON_BASE64=$(base64 -i gcp-storage-service-account.json) && \
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