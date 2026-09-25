#!/usr/bin/env python3
"""
Complete Member 2 training pipeline.
Loads training data, generates features, trains model, tunes threshold.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data_loader import load_train_sources, load_train_ground_truth
from .feature_engineering import features_from_pairs
from .threshold_tuning import fbeta_precision_recall
from .validation import validate_predictions, print_validation_report
from .config import cfg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def build_training_labels(features_df: pd.DataFrame, ground_truth: pd.DataFrame) -> np.ndarray:
    """
    Build binary labels for candidate pairs based on ground truth.
    label=1 if the candidate is in the matched set for that S1 entity, else 0.
    """
    # Build ground truth mapping
    gt_map = {}
    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]
        matched_ids = set([c.strip() for c in (row.get("matched_entity_ids") or "").split(",") if c.strip()])
        gt_map[s1_id] = matched_ids
    
    # Label each row
    y = []
    for _, row in features_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_id"]
        label = 1 if cand_id in gt_map.get(s1_id, set()) else 0
        y.append(label)
    
    return np.array(y)


def run_training(base_dir: str = ".", output_dir: str = None, model_path: str = None):
    """
    End-to-end training pipeline.
    """
    if output_dir is None:
        output_dir = os.path.join(base_dir, "output")
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "matcher.pkl")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    log.info("Loading training data...")
    s1, s2, s3 = load_train_sources()
    gt = load_train_ground_truth()
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    
    # Load or create candidate pairs for training
    train_cand_path = os.path.join(output_dir, "train_candidate_pairs.tsv")
    if os.path.isfile(train_cand_path):
        log.info(f"Loading training candidates from {train_cand_path}")
        candidates = pd.read_csv(train_cand_path, sep="\t", dtype=str)
    else:
        log.info("No train candidate pairs found; creating mock candidates from S2/S3 sample")
        # Simple heuristic: all S2 + all S3 as candidates for each S1
        # In production, this would come from Member 1's blocking on training data
        s2_ids = s2["entity_id"].tolist()
        s3_ids = s3["entity_id"].tolist()
        cand_list = s2_ids + s3_ids
        rows = []
        for s1_id in s1["entity_id"].tolist():
            rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cand_list)})
        candidates = pd.DataFrame(rows)
        log.warning("Using all S2+S3 as candidates (full recall, no blocking). In production, use Member 1's blocking.")
    
    log.info(f"Generating features for {len(candidates)} S1 entities and their candidates...")
    features = features_from_pairs(s1, candidates, s2s3)
    log.info(f"Generated {len(features)} candidate pair features")
    
    # Build labels
    log.info("Building training labels from ground truth...")
    y = build_training_labels(features, gt)
    log.info(f"Label distribution: {np.bincount(y)}")
    
    # Train model
    log.info("Training HistGradientBoostingClassifier...")
    X = features.drop(columns=["source1_entity_id", "candidate_id"])
    clf = HistGradientBoostingClassifier(random_state=cfg.seed, max_iter=100)
    clf.fit(X, y)
    
    # Get probabilities for threshold tuning
    probs = clf.predict_proba(X)[:, 1]
    features["prob"] = probs
    
    # Build true mapping
    true_map = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = set([c.strip() for c in (row.get("matched_entity_ids") or "").split(",") if c.strip()])
        true_map[s1_id] = matched
    
    # Tune threshold
    log.info("Tuning threshold on training data (validation)...")
    thresholds = list(np.linspace(0.1, 0.95, 18))
    best_f05 = -1
    best_threshold = 0.5
    best_metrics = None
    
    for t in thresholds:
        # Build predictions at this threshold
        pred_map = {}
        for s1_id in features["source1_entity_id"].unique():
            s1_rows = features[features["source1_entity_id"] == s1_id]
            matched = set(s1_rows[s1_rows["prob"] >= t]["candidate_id"].tolist())
            pred_map[s1_id] = matched
        
        # Evaluate
        val_results = validate_predictions(true_map, pred_map)
        f05 = val_results["macro_f05"]
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = t
            best_metrics = val_results
    
    log.info(f"Best threshold: {best_threshold:.3f}, F0.5: {best_f05:.4f}")
    print_validation_report(best_metrics)
    
    # Save model and threshold
    log.info(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    config = {
        "threshold": float(best_threshold),
        "best_f05": float(best_f05),
        "feature_names": X.columns.tolist(),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Saved config to {config_path}")
    
    log.info("Training complete!")
    return clf, best_threshold


if __name__ == "__main__":
    run_training()
