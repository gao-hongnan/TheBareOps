# Training Pipeline

## Setup Virtual Environment

```bash
cd pipeline-training && \
curl -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh && \
bash make_venv.sh venv --pyproject --dev && \
rm make_venv.sh && \
source venv/bin/activate
```

## Experiment Tracking

```bash
# https://stackoverflow.com/questions/69818376/localhost5000-unavailable-in-macos-v12-monterey
# mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri $PWD/stores/mlruns
cd pipeline-training && \
mlflow server --backend-store-uri $PWD/mlruns
pgrep -f mlflow | xargs kill
```


