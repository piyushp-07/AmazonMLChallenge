from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[4]

_default_data_root = REPO_ROOT / "student_resource"
if not (_default_data_root / "dataset").exists():
    _downloads_data_root = Path(
        r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
    )
    if (_downloads_data_root / "dataset").exists():
        _default_data_root = _downloads_data_root

DATA_ROOT = Path(
    __import__("os").environ.get(
        "AMON_DATA_ROOT",
        str(_default_data_root),
    )
).resolve()

TEST_DIR = DATA_ROOT / "dataset" / "test"

MODEL_PATH = (
    REPO_ROOT
    / "student_resource"
    / "models"
    / "matcher.pkl"
)

CONFIG_PATH = (
    REPO_ROOT
    / "student_resource"
    / "models"
    / "config.json"
)


# ============================================================
# Load Member 1 blocker
# ============================================================

MEMBER1_NORMALIZE = (
    REPO_ROOT
    / "src"
    / "preprocessing"
    / "normalize.py"
)

MEMBER1_BLOCKING = (
    REPO_ROOT
    / "src"
    / "blocking"
    / "candidate_generation.py"
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# Member 1 candidate_generation.py imports:
#
# from src.preprocessing.normalize import ...
#
# Create the compatibility package required by it.

import types

if "src.preprocessing" not in sys.modules:

    preprocessing_package = types.ModuleType(
        "src.preprocessing"
    )

    preprocessing_package.__path__ = [
        str(
            REPO_ROOT
            / "src"
            / "preprocessing"
        )
    ]

    sys.modules["src.preprocessing"] = (
        preprocessing_package
    )


normalize_module = load_module(
    "member1_normalize_e2e",
    MEMBER1_NORMALIZE,
)

sys.modules[
    "src.preprocessing.normalize"
] = normalize_module


blocking_module = load_module(
    "member1_blocking_e2e",
    MEMBER1_BLOCKING,
)

load_source = blocking_module.load_source
generate_candidates = blocking_module.generate_candidates


# ============================================================
# Load Member 2 feature engineering
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

PACKAGE_NAME = "e2e_business_entity_resolution"

package_module = types.ModuleType(
    PACKAGE_NAME
)

package_module.__path__ = [
    str(SRC_DIR)
]

sys.modules[PACKAGE_NAME] = package_module


FEATURE_PATH = SRC_DIR / "feature_engineering.py"

feature_module = load_module(
    f"{PACKAGE_NAME}.feature_engineering",
    FEATURE_PATH,
)

features_from_pairs = feature_module.features_from_pairs


# ============================================================
# Load test data
# ============================================================

print("=" * 70)
print("END-TO-END TEST")
print("=" * 70)

print("\nLoading test data...")

s1 = load_source(
    str(TEST_DIR / "test_source1.tsv")
)

s2 = load_source(
    str(TEST_DIR / "test_source2.tsv")
)

s3 = load_source(
    str(TEST_DIR / "test_source3.tsv")
)

print(f"Full S1 rows: {len(s1):,}")
print(f"Full S2 rows: {len(s2):,}")
print(f"Full S3 rows: {len(s3):,}")


# ============================================================
# Small S1 slice
# ============================================================

S1_SAMPLE_SIZE = 500

s1_sample = s1.head(
    S1_SAMPLE_SIZE
).copy()

print(
    f"\nUsing first {len(s1_sample):,} S1 entities."
)


# ============================================================
# Generate REAL Member 1 candidates
# ============================================================

print("\nGenerating S1 -> S2 candidates...")

s2_candidates = generate_candidates(
    s1_sample,
    s2,
)

print(
    f"S1 -> S2 candidates: "
    f"{len(s2_candidates):,}"
)


print("\nGenerating S1 -> S3 candidates...")

s3_candidates = generate_candidates(
    s1_sample,
    s3,
)

print(
    f"S1 -> S3 candidates: "
    f"{len(s3_candidates):,}"
)


# ============================================================
# Combine candidates
# ============================================================

s2_candidates = s2_candidates.copy()
s2_candidates["source"] = "source2"

s3_candidates = s3_candidates.copy()
s3_candidates["source"] = "source3"

candidates = pd.concat(
    [
        s2_candidates,
        s3_candidates,
    ],
    ignore_index=True,
)

candidates = candidates[
    [
        "source1_entity_id",
        "matched_entity_id",
        "source",
    ]
]

print(
    f"\nTotal real blocker candidates: "
    f"{len(candidates):,}"
)

print(
    f"Average candidates per S1: "
    f"{len(candidates) / len(s1_sample):,.2f}"
)


# ============================================================
# Convert to Member 2 feature format
# ============================================================

feature_candidates = candidates.rename(
    columns={
        "matched_entity_id": "candidate_id"
    }
)

print("\nGenerating matcher features...")

s2s3 = pd.concat(
    [
        s2,
        s3,
    ],
    ignore_index=True,
)

features = features_from_pairs(
    s1_sample,
    feature_candidates,
    s2s3,
)

print(
    f"Feature rows generated: "
    f"{len(features):,}"
)


if features.empty:

    print(
        "\nNo candidate features were generated."
    )

    raise SystemExit(1)


# ============================================================
# Load trained model
# ============================================================

print("\nLoading trained matcher...")

with open(
    MODEL_PATH,
    "rb",
) as f:

    model = pickle.load(f)


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    config = json.load(f)


threshold = float(
    config["threshold"]
)

print(
    f"Model threshold: {threshold:.2f}"
)


# ============================================================
# Score candidates
# ============================================================

X = features.drop(
    columns=[
        "source1_entity_id",
        "candidate_id",
    ]
)

print(
    "\nScoring candidates..."
)

probabilities = model.predict_proba(
    X
)[:, 1]

features["probability"] = probabilities


# ============================================================
# Build predictions
# ============================================================

predictions = {}

for s1_id, group in features.groupby(
    "source1_entity_id"
):

    matched_ids = (
        group.loc[
            group["probability"] >= threshold,
            "candidate_id",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    predictions[str(s1_id)] = sorted(
        matched_ids
    )


# Ensure every sampled S1 exists.

for s1_id in s1_sample["entity_id"].astype(str):

    if s1_id not in predictions:
        predictions[s1_id] = []


# ============================================================
# Summary
# ============================================================

entities_with_matches = sum(
    1
    for ids in predictions.values()
    if ids
)

singletons = (
    len(predictions)
    - entities_with_matches
)

total_matches = sum(
    len(ids)
    for ids in predictions.values()
)

print("\n" + "=" * 70)
print("END-TO-END RESULT")
print("=" * 70)

print(
    f"S1 entities tested: "
    f"{len(predictions):,}"
)

print(
    f"Entities with predicted matches: "
    f"{entities_with_matches:,}"
)

print(
    f"Predicted singletons: "
    f"{singletons:,}"
)

print(
    f"Total predicted links: "
    f"{total_matches:,}"
)

print(
    f"Average predicted links/S1: "
    f"{total_matches / len(predictions):.2f}"
)

print(
    f"Threshold: "
    f"{threshold:.2f}"
)

print("=" * 70)

print(
    "\nEnd-to-end test completed successfully."
)
