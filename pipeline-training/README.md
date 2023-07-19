# Training Pipeline

## MLOps (Training Pipeline)

1. **Scheduled Jobs (CronJobs in Kubernetes):** This can be a good option if
   your model needs to be retrained periodically on a regular basis (for
   example, daily or weekly), especially if the training data changes
   frequently. The downside is that the training might run even if the model or
   data hasn't significantly changed, which might be an unnecessary use of
   resources.

2. **Event-driven Approach:** In this scenario, training is triggered by
   specific events, such as changes to the data or to the model code. This can
   be more efficient because it ensures that training only happens when
   necessary. However, it might be more complex to set up, depending on your
   infrastructure.

3. **Continuous Training:** In some cases, you might want to train your model
   continuously, for example in online learning scenarios or when using
   reinforcement learning. This might require a more complex setup and more
   computational resources.

4. **Manual Trigger:** In some cases, you might want to control when the
   training happens, for example when you're still in the process of developing
   and testing your model. In this case, you might want to set up your pipeline
   so that you can manually trigger the training.

5. **Automatic Model Retraining:** Some advanced MLOps setups use monitoring and
   automated processes to retrain models when the model performance drops below
   a certain level or when drift in the data is detected. This can be complex to
   set up but might lead to better and more timely model updates.

In summary, the decision on how to set up your training pipeline depends on the
specifics of your project, such as how often your data changes, how long
training takes, how much computational resources you have, and how critical it
is to update your model quickly when changes occur.

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
mlflow server --backend-store-uri $PWD/mlruns
pgrep -f mlflow | xargs kill
```
