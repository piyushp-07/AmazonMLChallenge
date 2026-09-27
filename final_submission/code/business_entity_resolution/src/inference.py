#!/usr/bin/env python3
"""Chunked inference with the repository's trained matcher."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import gc
from pathlib import Path

import joblib
import pandas as pd

from .config import cfg
from .data_loader import load_test_sources
from .feature_engineering import features_from_pairs

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def run_inference(
    candidate_path: str | None = None,
    test_dir: str | None = None,
    model_path: str | None = None,
    config_path: str | None = None,
    output_path: str | None = None,
):
    candidate_path = Path(candidate_path or cfg.candidate_path)
    test_dir = Path(test_dir or cfg.test_dir)
    model_path = Path(model_path or cfg.model_path)
    config_path = Path(config_path or model_path.parent / "config.json")
    output_path = Path(output_path or cfg.matching_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for path, label in ((candidate_path, "Candidate file"), (model_path, "Model"), (config_path, "Model configuration")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    header = list(pd.read_csv(candidate_path, sep="\t", nrows=0).columns)
    expected = ["source1_entity_id", "candidate_entity_ids"]
    if header != expected:
        raise ValueError(f"Candidate schema must be {expected}, got {header}")

    log.info("Loading test data...")
    s1, s2, s3 = load_test_sources(test_dir)
    log.info(f"Test rows: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")
    s2s3 = pd.concat([s2, s3], ignore_index=True)

    # Retain the supplied data and model; build the record maps only once.
    log.info("Indexing test records for candidate lookups...")
    s1_lookup = s1.set_index("entity_id").to_dict(orient="index")
    candidate_lookup = s2s3.set_index("entity_id").to_dict(orient="index")
    s1_ids = s1["entity_id"].astype(str).tolist()
    del s1, s2, s3, s2s3
    gc.collect()
    empty_frame = pd.DataFrame()

    model = joblib.load(model_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "threshold" not in config:
        raise KeyError("Model configuration does not contain 'threshold'.")
    threshold = float(config["threshold"])
    log.info(f"Using supplied threshold: {threshold:.3f}")

    predictions: dict[str, set[str]] = {}
    scored_pairs = 0
    candidate_rows = 0
    for candidates in pd.read_csv(
        candidate_path, sep="\t", dtype=str, keep_default_na=False, chunksize=2000
    ):
        candidate_rows += len(candidates)
        features = features_from_pairs(
            empty_frame, candidates, empty_frame,
            s1_lookup=s1_lookup,
            candidate_lookup=candidate_lookup,
        )
        if features.empty:
            continue
        feature_columns = [
            column for column in features.columns
            if column not in {"source1_entity_id", "candidate_id"}
        ]
        probabilities = model.predict_proba(features[feature_columns])[:, 1]
        scored_pairs += len(features)
        accepted = features.loc[
            probabilities >= threshold,
            ["source1_entity_id", "candidate_id"],
        ]
        for s1_id, candidate_id in accepted.itertuples(index=False, name=None):
            predictions.setdefault(str(s1_id), set()).add(str(candidate_id))
        if candidate_rows % 100_000 < 2_000:
            log.info(f"Scored candidate rows for {candidate_rows:,} Source 1 entities")

    if scored_pairs == 0:
        raise RuntimeError("No candidate pairs were scored; refusing to write predictions.")

    temp_output = output_path.with_name(output_path.name + ".tmp")
    with temp_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in s1_ids:
            writer.writerow([s1_id, ",".join(sorted(predictions.get(s1_id, set())))])
    temp_output.replace(output_path)

    log.info(f"Candidate rows: {candidate_rows:,}")
    log.info(f"Candidate pairs scored: {scored_pairs:,}")
    log.info(f"S1 entities with matches: {len(predictions):,}")
    log.info(f"Total matched IDs: {sum(len(ids) for ids in predictions.values()):,}")
    log.info(f"Wrote matching results: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score test candidates with the supplied trained matcher.")
    parser.add_argument("--candidate", default=None, help="Candidate TSV path")
    parser.add_argument("--test-dir", default=None, help="Test data directory")
    parser.add_argument("--model", default=None, help="Trained matcher pickle")
    parser.add_argument("--config", default=None, help="Matcher configuration JSON")
    parser.add_argument("--output", default=None, help="Matching TSV output path")
    args = parser.parse_args()
    run_inference(args.candidate, args.test_dir, args.model, args.config, args.output)
