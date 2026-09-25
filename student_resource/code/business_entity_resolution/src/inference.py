#!/usr/bin/env python3
"""
Member 2 inference pipeline.
Loads test data, applies trained model to candidates, produces matching_results.tsv.
"""

import os
import json
import pickle
import logging
import pandas as pd

from .data_loader import load_test_sources
from .feature_engineering import features_from_pairs
from .config import cfg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_inference(
    candidate_path: str = None,
    test_dir: str = None,
    model_path: str = None,
    config_path: str = None,
    output_path: str = None,
):
    """
    Complete inference pipeline.
    
    Args:
        candidate_path: path to candidate_pairs.tsv from Member 1 blocking
        test_dir: path to test dataset directory
        model_path: path to trained model pickle
        config_path: path to config.json (contains threshold)
        output_path: path to write matching_results.tsv
    """
    if candidate_path is None:
        candidate_path = cfg.candidate_path
    if test_dir is None:
        test_dir = cfg.test_dir
    if model_path is None:
        model_path = cfg.model_path
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if output_path is None:
        output_path = cfg.matching_out
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load test data
    log.info("Loading test data...")
    s1, s2, s3 = load_test_sources(test_dir)
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    log.info(f"Test: {len(s1)} S1, {len(s2)} S2, {len(s3)} S3 entities")
    
    # Load candidates
    log.info(f"Loading candidate pairs from {candidate_path}")
    if not os.path.isfile(candidate_path):
        log.error(f"Candidate file not found: {candidate_path}")
        log.warning("Creating output with all S1 entities as singletons (no matches)")
        out = pd.DataFrame({
            "source1_entity_id": s1["entity_id"].tolist(),
            "matched_entity_ids": "",
        })
        out.to_csv(output_path, sep="\t", index=False)
        log.info(f"Wrote placeholder output to {output_path}")
        return
    
    candidates = pd.read_csv(candidate_path, sep="\t", dtype=str)
    log.info(f"Loaded candidates for {len(candidates)} S1 entities")
    
    # Generate features
    log.info("Generating features for test candidates...")
    features = features_from_pairs(s1, candidates, s2s3)
    if len(features) == 0:
        log.warning("No features generated; all S1 entities will be singletons")
        out = pd.DataFrame({
            "source1_entity_id": s1["entity_id"].tolist(),
            "matched_entity_ids": "",
        })
        out.to_csv(output_path, sep="\t", index=False)
        log.info(f"Wrote output to {output_path}")
        return
    
    log.info(f"Generated {len(features)} candidate pair features")
    
    # Load model and config
    log.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    
    log.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    threshold = config["threshold"]
    log.info(f"Using threshold: {threshold:.3f}")
    
    # Score candidates
    X = features.drop(columns=["source1_entity_id", "candidate_id"])
    probs = clf.predict_proba(X)[:, 1]
    features["prob"] = probs
    
    # Build predictions
    log.info(f"Applying threshold {threshold:.3f} to get final matches...")
    predictions = {}
    for s1_id in features["source1_entity_id"].unique():
        s1_rows = features[features["source1_entity_id"] == s1_id]
        matched_ids = s1_rows[s1_rows["prob"] >= threshold]["candidate_id"].tolist()
        # Remove duplicates and sort
        matched_ids = sorted(list(set(matched_ids)))
        predictions[s1_id] = matched_ids
    
    # Ensure all S1 entities are in predictions
    for s1_id in s1["entity_id"].tolist():
        if s1_id not in predictions:
            predictions[s1_id] = []
    
    # Write output
    log.info(f"Writing matching results to {output_path}")
    rows = []
    for s1_id in sorted(s1["entity_id"].tolist()):
        matched_ids = ",".join(predictions.get(s1_id, []))
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_ids})
    
    out = pd.DataFrame(rows)
    out.to_csv(output_path, sep="\t", index=False)
    log.info(f"Wrote {len(out)} rows to {output_path}")
    
    # Print summary
    num_matches = sum(1 for ids in predictions.values() if ids)
    num_singletons = len(predictions) - num_matches
    total_matched = sum(len(ids) for ids in predictions.values())
    print("\n" + "="*70)
    print("INFERENCE SUMMARY")
    print("="*70)
    print(f"Total S1 entities: {len(predictions)}")
    print(f"Singletons (no match): {num_singletons}")
    print(f"Entities with matches: {num_matches}")
    print(f"Total matched IDs: {total_matched}")
    print(f"Threshold used: {threshold:.3f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_inference()
