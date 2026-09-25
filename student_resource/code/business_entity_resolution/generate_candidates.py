#!/usr/bin/env python3
"""
Wrapper script to run Member 1's blocking on full test data and format the output.
Generates candidate_pairs.tsv in the required format for Member 2.
"""

import sys
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Add member1 repo to path
MEMBER1_REPO = "/home/piyush/VS/AmazonMLChallenge/member1_repo"
STUDENT_RESOURCE = "/home/piyush/VS/AmazonMLChallenge/student_resource"

sys.path.insert(0, MEMBER1_REPO)

from src.blocking.candidate_generation import load_source, generate_candidates


def run_member1_blocking(test_or_train: str = "test", output_path: str = None):
    """
    Run Member 1's blocking to generate candidates for test/train data.
    
    Args:
        test_or_train: "test" or "train"
        output_path: where to save candidate_pairs.tsv
    
    Returns:
        DataFrame with columns: source1_entity_id, candidate_entity_ids (comma-separated)
    """
    if test_or_train == "test":
        dataset_dir = os.path.join(STUDENT_RESOURCE, "dataset", "test")
    else:
        dataset_dir = os.path.join(STUDENT_RESOURCE, "dataset", "train")
    
    log.info(f"Loading {test_or_train.upper()} data from {dataset_dir}...")
    
    s1_path = os.path.join(dataset_dir, f"{test_or_train}_source1.tsv")
    s2_path = os.path.join(dataset_dir, f"{test_or_train}_source2.tsv")
    s3_path = os.path.join(dataset_dir, f"{test_or_train}_source3.tsv")
    
    # Load and normalize data
    log.info("Loading Source 1...")
    source1 = load_source(s1_path)
    log.info(f"  Loaded {len(source1):,} S1 records")
    
    log.info("Loading Source 2...")
    source2 = load_source(s2_path)
    log.info(f"  Loaded {len(source2):,} S2 records")
    
    log.info("Loading Source 3...")
    source3 = load_source(s3_path)
    log.info(f"  Loaded {len(source3):,} S3 records")
    
    # Generate candidates: S1 → S2
    log.info("\nGenerating candidates for S1 -> S2...")
    candidates_s2 = generate_candidates(source1, source2)
    log.info(f"  Generated {len(candidates_s2):,} S1-S2 candidate pairs")
    
    # Generate candidates: S1 → S3
    log.info("Generating candidates for S1 -> S3...")
    candidates_s3 = generate_candidates(source1, source3)
    log.info(f"  Generated {len(candidates_s3):,} S1-S3 candidate pairs")
    
    # Combine S2 and S3 candidates
    log.info("\nCombining S2 and S3 candidates...")
    all_candidates = pd.concat([candidates_s2, candidates_s3], ignore_index=True)
    log.info(f"  Total: {len(all_candidates):,} candidate pairs")
    
    # Rename column and aggregate by S1 entity
    all_candidates = all_candidates.rename(columns={"matched_entity_id": "candidate_entity_id"})
    
    # Group by S1 and collect all candidates
    grouped = all_candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(sorted(set(ids)))
    ).reset_index()
    grouped.columns = ["source1_entity_id", "candidate_entity_ids"]
    
    log.info(f"Aggregated into {len(grouped):,} S1 entities with candidates")
    
    # Ensure all S1 entities are in output (even those with no candidates)
    all_s1_ids = set(source1["entity_id"].tolist())
    grouped_s1_ids = set(grouped["source1_entity_id"].tolist())
    missing_s1 = all_s1_ids - grouped_s1_ids
    
    if missing_s1:
        log.info(f"Adding {len(missing_s1):,} S1 entities with no candidates...")
        missing_rows = pd.DataFrame({
            "source1_entity_id": sorted(missing_s1),
            "candidate_entity_ids": ""
        })
        grouped = pd.concat([grouped, missing_rows], ignore_index=True)
    
    # Sort by S1 ID
    grouped = grouped.sort_values("source1_entity_id").reset_index(drop=True)
    
    # Write to file
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        log.info(f"\nWriting candidates to {output_path}...")
        grouped.to_csv(output_path, sep="\t", index=False)
        log.info(f"  Wrote {len(grouped):,} rows")
    
    return grouped


if __name__ == "__main__":
    # Generate test candidates
    test_output = os.path.join(STUDENT_RESOURCE, "output", "candidate_pairs.tsv")
    test_cands = run_member1_blocking(test_or_train="test", output_path=test_output)
    
    print("\n" + "="*70)
    print("Sample test candidates (first 5 rows):")
    print("="*70)
    print(test_cands.head().to_string(index=False))
    print("\n")
