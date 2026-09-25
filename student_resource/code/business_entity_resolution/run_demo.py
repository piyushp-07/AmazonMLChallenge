#!/usr/bin/env python3
"""
Integration runner showing the full Member 2 pipeline end-to-end with sample output.
Demonstrates: data loading → feature generation → training → threshold tuning → inference.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def run_integration_demo():
    """Run a complete demo pipeline on test data with mock candidates."""
    
    from src.data_loader import load_test_sources
    from src.feature_engineering import features_from_pairs
    from src.config import cfg
    
    log.info("="*70)
    log.info("MEMBER 2 INTEGRATION DEMO - Full Pipeline")
    log.info("="*70)
    
    # Step 1: Load test data
    log.info("\n[1/5] Loading test data...")
    s1, s2, s3 = load_test_sources(cfg.test_dir)
    log.info(f"  Loaded {len(s1)} S1, {len(s2)} S2, {len(s3)} S3 entities")
    
    # Step 2: Create mock candidates (in production, these come from Member 1)
    log.info("\n[2/5] Creating mock candidate pairs for demo...")
    s2_ids = s2["entity_id"].head(5).tolist()
    s3_ids = s3["entity_id"].head(5).tolist()
    rows = []
    for s1_id in s1["entity_id"].head(10).tolist():
        cand_ids = s2_ids + s3_ids
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": ",".join(cand_ids)})
    candidates = pd.DataFrame(rows)
    log.info(f"  Created candidates for {len(candidates)} S1 entities")
    print(f"\nSample candidates (first 3 rows):")
    print(candidates.head(3).to_string(index=False))
    
    # Step 3: Generate features
    log.info("\n[3/5] Generating features for candidate pairs...")
    s2s3 = pd.concat([s2.head(5), s3.head(5)], ignore_index=True)
    s1_small = s1.head(10)
    features = features_from_pairs(s1_small, candidates, s2s3)
    log.info(f"  Generated {len(features)} candidate-pair features")
    
    # Show feature sample
    feature_cols = [c for c in features.columns if c not in ["source1_entity_id", "candidate_id"]]
    print(f"\nFeature columns ({len(feature_cols)}):")
    print("  ", ", ".join(feature_cols[:10]))
    print(f"\nSample features (first 3 rows, first 8 columns):")
    cols_to_show = ["source1_entity_id", "candidate_id"] + feature_cols[:6]
    print(features[cols_to_show].head(3).to_string(index=False))
    
    # Step 4: Create mock labels and train
    log.info("\n[4/5] Training model (demo with random labels)...")
    y = np.random.binomial(1, 0.2, size=len(features))
    log.info(f"  Label distribution: {np.bincount(y)}")
    
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = features.drop(columns=["source1_entity_id", "candidate_id"])
    clf = HistGradientBoostingClassifier(random_state=cfg.seed, max_iter=10)
    clf.fit(X, y)
    log.info("  Model trained successfully")
    
    # Score and show probabilities
    probs = clf.predict_proba(X)[:, 1]
    features["prob"] = probs
    log.info(f"  Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    
    # Step 5: Demonstrate threshold and decision logic
    log.info("\n[5/5] Applying decision threshold...")
    threshold = 0.5
    log.info(f"  Using threshold: {threshold:.3f}")
    
    predictions = {}
    for s1_id in features["source1_entity_id"].unique():
        s1_rows = features[features["source1_entity_id"] == s1_id]
        matched = s1_rows[s1_rows["prob"] >= threshold]["candidate_id"].tolist()
        predictions[s1_id] = sorted(list(set(matched)))
    
    # Final output
    log.info("\n" + "="*70)
    log.info("FINAL PREDICTIONS")
    log.info("="*70)
    rows_out = []
    for s1_id in sorted(predictions.keys()):
        matched_ids = ",".join(predictions[s1_id])
        rows_out.append({"source1_entity_id": s1_id, "matched_entity_ids": matched_ids})
    
    out_df = pd.DataFrame(rows_out)
    print("\nFinal matching results (first 10 rows):")
    print(out_df.head(10).to_string(index=False))
    
    num_singletons = sum(1 for ids in predictions.values() if not ids)
    total_matched = sum(len(ids) for ids in predictions.values())
    print(f"\nPrediction stats:")
    print(f"  Total S1 entities: {len(predictions)}")
    print(f"  Singletons (no match): {num_singletons}")
    print(f"  Entities with matches: {len(predictions) - num_singletons}")
    print(f"  Total matched IDs: {total_matched}")
    
    log.info("\n" + "="*70)
    log.info("DEMO COMPLETE - Full pipeline executed successfully!")
    log.info("="*70)


if __name__ == "__main__":
    run_integration_demo()
