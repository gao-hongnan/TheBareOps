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