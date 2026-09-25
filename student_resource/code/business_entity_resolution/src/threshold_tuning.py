import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def fbeta_precision_recall(p_true_lists, pred_lists, beta=0.5):
    # p_true_lists and pred_lists are dicts: s1_id -> set(matched ids)
    per_entity = []
    for s1, true_set in p_true_lists.items():
        pred_set = pred_lists.get(s1, set())
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        precision = 1.0 if tp + fp == 0 and len(pred_set) == 0 else (tp / (tp + fp) if tp + fp > 0 else 0.0)
        recall = 1.0 if tp + fn == 0 and len(true_set) == 0 else (tp / (tp + fn) if tp + fn > 0 else 0.0)
        if precision == 0 and recall == 0:
            fbeta = 0.0
        else:
            fbeta = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall) if (beta * beta * precision + recall) > 0 else 0.0
        per_entity.append((precision, recall, fbeta))
    per_entity = np.array(per_entity)
    return per_entity[:, 0].mean(), per_entity[:, 1].mean(), per_entity[:, 2].mean()


def tune_threshold(probs_df, true_mapping, thresholds=None):
    # probs_df: rows with source1_entity_id, candidate_id, prob
    if thresholds is None:
        thresholds = list(np.linspace(0.1, 0.95, 18))
    best = (0.0, 0.0, 0.0, 0.5)
    grouped = probs_df.groupby("source1_entity_id")
    for t in thresholds:
        pred = {}
        for s1, g in grouped:
            sel = g[g["prob"] >= t]["candidate_id"].tolist()
            pred[s1] = set(sel)
        p, r, f = fbeta_precision_recall(true_mapping, pred, beta=0.5)
        if f > best[0]:
            best = (f, p, r, t)
    return {"best_f": best[0], "precision": best[1], "recall": best[2], "threshold": best[3]}
