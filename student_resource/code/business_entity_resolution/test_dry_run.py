#!/usr/bin/env python3
"""
Dry-run test for the matching pipeline.

Uses a tiny slice of the test data (20 rows per source) and
mock candidate pairs to verify that data loading and feature
engineering work end-to-end without running the full pipeline.

Safe to run at any time — no model loading, no file writes,
no expensive blocking over the full dataset.

Run from the repository root:
    python -m student_resource.code.business_entity_resolution.test_dry_run

Or from the business_entity_resolution directory:
    python test_dry_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package is importable regardless of working directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd

from src.data_loader import load_test_sources
from src.feature_engineering import features_from_pairs


def build_mock_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    n_s1: int = 10,
    k: int = 3,
) -> pd.DataFrame:
    """
    Build mock candidate pairs without running real blocking.

    Each of the first n_s1 S1 entities is paired with the first
    k S2 entities and the first k S3 entities.
    """
    s2_ids = s2["entity_id"].astype(str).tolist()[:k]
    s3_ids = s3["entity_id"].astype(str).tolist()[:k]
    cand_ids = ",".join(s2_ids + s3_ids)

    rows = [
        {
            "source1_entity_id": eid,
            "candidate_entity_ids": cand_ids,
        }
        for eid in s1["entity_id"].astype(str).tolist()[:n_s1]
    ]

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("DRY-RUN: data loading + feature engineering")
    print("=" * 60)

    # Load only the first 20 rows of each source.
    print("\nLoading test sources (head 20)...")
    s1_full, s2_full, s3_full = load_test_sources()

    s1 = s1_full.head(20).copy()
    s2 = s2_full.head(20).copy()
    s3 = s3_full.head(20).copy()

    print(f"  S1: {len(s1)} rows | S2: {len(s2)} rows | S3: {len(s3)} rows")

    # Build mock candidates (no real blocking).
    print("\nBuilding mock candidates (n_s1=10, k=3)...")
    candidates = build_mock_candidates(s1, s2, s3, n_s1=10, k=3)
    print(f"  Candidate rows: {len(candidates)}")
    print(candidates.to_string(index=False))

    # Run feature engineering.
    print("\nRunning feature engineering...")
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    features = features_from_pairs(s1, candidates, s2s3)

    print(f"\nFeatures shape: {features.shape}")
    print("\nFirst 5 rows:")
    print(features.head().to_string())

    print("\n" + "=" * 60)
    print("DRY-RUN PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
