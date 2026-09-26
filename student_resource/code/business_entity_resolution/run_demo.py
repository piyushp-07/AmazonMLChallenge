from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import cfg
from src.feature_engineering import features_from_pairs


def load_sample(path: Path, nrows: int) -> pd.DataFrame:
    """Load only a small sample from a TSV file."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows,
    )


def build_mock_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    candidates_per_s1: int = 3,
) -> pd.DataFrame:
    """Create a tiny deterministic candidate set for demonstration."""

    source_rows = pd.concat(
        [
            source2.assign(_source="source2"),
            source3.assign(_source="source3"),
        ],
        ignore_index=True,
    )

    records = []

    for _, s1_row in source1.iterrows():
        selected = source_rows.head(candidates_per_s1)

        for _, candidate in selected.iterrows():
            records.append(
                {
                    "source1_entity_id": s1_row["entity_id"],
                    "candidate_id": candidate["entity_id"],
                }
            )

    return pd.DataFrame(records)


def run_demo(nrows: int = 20, candidates_per_s1: int = 3) -> None:
    print("=" * 70)
    print("AMON LIGHTWEIGHT DEMO")
    print("=" * 70)

    test_dir = cfg.test_dir

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    print(f"Loading at most {nrows} rows per source...")

    s1 = load_sample(s1_path, nrows)
    s2 = load_sample(s2_path, nrows)
    s3 = load_sample(s3_path, nrows)

    print(
        f"S1: {len(s1)} rows | "
        f"S2: {len(s2)} rows | "
        f"S3: {len(s3)} rows"
    )

    print("")
    print(
        f"Building mock candidates "
        f"(n_s1={len(s1)}, k={candidates_per_s1})..."
    )

    candidates = build_mock_candidates(
        s1,
        s2,
        s3,
        candidates_per_s1,
    )

    print(f"Mock candidate pairs: {len(candidates)}")

    print("")
    print("Running feature engineering...")

    source2_features = s2[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    source3_features = s3[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    all_candidates = pd.concat(
        [
            source2_features,
            source3_features,
        ],
        ignore_index=True,
    )

    features = features_from_pairs(
        s1,
        candidates,
        all_candidates,
    )

    print(f"Features shape: {features.shape}")

    expected_columns = {
        "source1_entity_id",
        "candidate_id",
    }

    if not expected_columns.issubset(features.columns):
        raise RuntimeError(
            "Feature engineering output is missing required ID columns."
        )

    feature_columns = [
        column
        for column in features.columns
        if column not in expected_columns
    ]

    if len(feature_columns) != 21:
        raise RuntimeError(
            f"Expected 21 model features, found {len(feature_columns)}."
        )

    print(f"Model feature count: {len(feature_columns)}")
    print("")
    print("=" * 70)
    print("LIGHTWEIGHT DEMO PASSED")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a lightweight AMON pipeline demonstration."
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=20,
        help="Rows loaded from each source. Default: 20.",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=3,
        help="Mock candidates per S1. Default: 3.",
    )

    args = parser.parse_args()

    if args.rows <= 0:
        parser.error("--rows must be greater than 0.")

    if args.candidates <= 0:
        parser.error("--candidates must be greater than 0.")

    run_demo(
        nrows=args.rows,
        candidates_per_s1=args.candidates,
    )


if __name__ == "__main__":
    main()
