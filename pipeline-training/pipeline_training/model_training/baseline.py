import pickle
import warnings
from typing import Any, Dict

import mlflow
import numpy as np
from common_utils.core.common import seed_all
from rich.pretty import pprint
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier  # pylint: disable=unused-import
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from pipeline_training.utils.common import log_data_splits_summary
