# Serving Pipeline

See my mlops docs for details, but here is concrete implementation
so we need this as well for case study.

## Setup Virtual Environment

```bash
cd pipeline-serving && \
curl -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh && \
bash make_venv.sh venv --pyproject --dev && \
rm make_venv.sh && \
source venv/bin/activate
```

uvicorn api:app --reload

{
  "data": [
    {
      "open": 0,
      "high": 0,
      "low": 0,
      "volume": 0,
      "number_of_trades": 0,
      "taker_buy_base_asset_volume": 0,
      "taker_buy_quote_asset_volume": 0
    }
  ]
}

## CICD

Seems like can depend on?

```
deploy_model:
  needs: [train_model]
  runs-on: ubuntu-latest
```