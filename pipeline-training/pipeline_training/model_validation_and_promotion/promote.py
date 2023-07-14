"""See https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
model validation section.

This script handles promoting a model to production.
Simple logic is if the new model is better than the current production model, then promote it to production.
Of course in real life, you might want to implement more robust model comparison logic and
implement a manual approval step before promoting the model to production. Include A/B testing too.
"""

from common_utils.experiment_tracking.promoter.core import MLFlowPromotionManager
