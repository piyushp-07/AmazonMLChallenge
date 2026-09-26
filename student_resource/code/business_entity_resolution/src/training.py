import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from .config import cfg


def train_model(X: pd.DataFrame, y: pd.Series, model_path: str = cfg.model_path):
    # simple baseline: HistGradientBoosting
    clf = HistGradientBoostingClassifier(random_state=cfg.seed)
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    return clf


def load_model(path: str = cfg.model_path):
    import joblib

    return joblib.load(path)
