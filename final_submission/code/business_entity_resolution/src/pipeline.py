import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

from .config import cfg
from .feature_engineering import features_from_pairs
from .training import train_model


# ============================================================
# Member 1 integration
# ============================================================
#
# Member 1's code lives in:
#   C:\Users\shiva\AmazonMLChallenge\src
#
# Member 2's code lives in:
#   C:\Users\shiva\AmazonMLChallenge\student_resource\
#   code\business_entity_resolution\src
#
# Both projects use the package name "src", so we must NOT
# replace sys.modules["src"]. We only provide the specific
# compatibility module required by Member 1's blocker.
#

MEMBER1_ROOT = Path(__file__).resolve().parents[4]

member1_src = MEMBER1_ROOT / "src"
member1_blocking = member1_src / "blocking"
member1_preprocessing = member1_src / "preprocessing"


def _load_module(module_name: str, file_path: Path):
    """Load a Python module directly from a file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Member 1 file not found: {file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# ------------------------------------------------------------
# Load Member 1 normalization
# ------------------------------------------------------------

member1_normalize = _load_module(
    "member1_normalize",
    member1_preprocessing / "normalize.py",
)


# ------------------------------------------------------------
# Compatibility package for Member 1 candidate_generation.py
# ------------------------------------------------------------
#
# Member 1 candidate_generation.py contains:
#
#     from src.preprocessing.normalize import ...
#
# We keep Member 2's "src" package untouched and create only
# the missing "src.preprocessing" package.
#

if "src.preprocessing" not in sys.modules:
    member1_preprocessing_package = types.ModuleType(
        "src.preprocessing"
    )
    member1_preprocessing_package.__path__ = [
        str(member1_preprocessing)
    ]
    sys.modules["src.preprocessing"] = member1_preprocessing_package


sys.modules["src.preprocessing.normalize"] = member1_normalize


# ------------------------------------------------------------
# Load Member 1 candidate generation
# ------------------------------------------------------------

member1_blocking_module = _load_module(
    "member1_candidate_generation",
    member1_blocking / "candidate_generation.py",
)


generate_candidates = member1_blocking_module.generate_candidates
load_source = member1_blocking_module.load_source


# ============================================================
# Candidate generation
# ============================================================

def _combine_candidates(source1_df, source2_df, source3_df):
    """
    Generate Member 1 candidates for S1->S2 and S1->S3,
    then combine them into the challenge candidate format.
    """

    s2_pairs = generate_candidates(
        source1_df,
        source2_df,
    ).rename(
        columns={
            "matched_entity_id": "candidate_entity_id"
        }
    )

    s3_pairs = generate_candidates(
        source1_df,
        source3_df,
    ).rename(
        columns={
            "matched_entity_id": "candidate_entity_id"
        }
    )

    combined = pd.concat(
        [s2_pairs, s3_pairs],
        ignore_index=True,
    )

    combined = (
        combined
        .groupby("source1_entity_id")["candidate_entity_id"]
        .agg(
            lambda ids: ",".join(
                sorted(set(ids))
            )
        )
        .reset_index()
        .rename(
            columns={
                "candidate_entity_id": "candidate_entity_ids"
            }
        )
    )

    return combined


def generate_candidate_pairs(train_or_test: str = "train"):
    """
    Generate candidate_entity_ids for every S1 entity.

    Returns:
        DataFrame with:
            source1_entity_id
            candidate_entity_ids
    """

    if train_or_test == "train":

        s1 = load_source(
            str(cfg.train_dir / "train_source1.tsv")
        )

        s2 = load_source(
            str(cfg.train_dir / "train_source2.tsv")
        )

        s3 = load_source(
            str(cfg.train_dir / "train_source3.tsv")
        )

    elif train_or_test == "test":

        s1 = load_source(
            str(cfg.test_dir / "test_source1.tsv")
        )

        s2 = load_source(
            str(cfg.test_dir / "test_source2.tsv")
        )

        s3 = load_source(
            str(cfg.test_dir / "test_source3.tsv")
        )

    else:
        raise ValueError(
            "train_or_test must be either 'train' or 'test'"
        )

    candidates = _combine_candidates(
        s1,
        s2,
        s3,
    )

    # Make sure every S1 entity gets a row.
    all_s1 = pd.read_csv(
        cfg.train_dir / "train_source1.tsv"
        if train_or_test == "train"
        else cfg.test_dir / "test_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id"],
    )

    all_s1 = all_s1.rename(
        columns={
            "entity_id": "source1_entity_id"
        }
    )

    candidates = all_s1.merge(
        candidates,
        how="left",
        on="source1_entity_id",
    )

    candidates["candidate_entity_ids"] = (
        candidates["candidate_entity_ids"]
        .fillna("")
    )

    candidates["source1_entity_id"] = (
        candidates["source1_entity_id"]
        .astype(str)
    )

    candidates = candidates[
        [
            "source1_entity_id",
            "candidate_entity_ids",
        ]
    ]

    return candidates


# ============================================================
# Training feature generation
# ============================================================

def build_training_features(train_candidates: pd.DataFrame):
    """
    Convert candidate IDs into Member 2 matching features.
    """

    from .data_loader import load_train_sources

    s1, s2, s3 = load_train_sources()

    s2s3 = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    return features_from_pairs(
        s1,
        train_candidates,
        s2s3,
    )


# ============================================================
# Training pipeline
# ============================================================

def main():

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("AMON Business Entity Resolution - Member 2 Pipeline")
    print("=" * 70)

    print("\nGenerating training candidates...")

    train_candidates = generate_candidate_pairs(
        "train"
    )

    train_candidate_path = (
        cfg.output_dir / "train_candidate_pairs.tsv"
    )

    train_candidates.to_csv(
        train_candidate_path,
        sep="\t",
        index=False,
    )

    print(
        "Training candidates written to:",
        train_candidate_path,
    )

    print(
        "Number of S1 entities:",
        len(train_candidates),
    )

    print(
        "Total candidate IDs:",
        train_candidates["candidate_entity_ids"]
        .str.split(",")
        .str.len()
        .sum(),
    )

    print("\nLoading ground truth...")

    from .data_loader import load_train_ground_truth

    gt = load_train_ground_truth()

    gt_map = {}

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        value = row["matched_entity_ids"]

        if value:
            ids = {
                item.strip()
                for item in value.split(",")
                if item.strip()
            }
        else:
            ids = set()

        gt_map[s1_id] = ids

    print(
        "Ground-truth S1 entities:",
        len(gt_map),
    )

    print("\nGenerating training features...")

    feats = build_training_features(
        train_candidates
    )

    print(
        "Feature rows:",
        len(feats),
    )

    if feats.empty:
        print(
            "ERROR: No training candidates/features were generated."
        )
        return

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    print("\nCreating training labels...")

    labels = []

    for _, row in feats.iterrows():

        s1_id = row["source1_entity_id"]
        candidate_id = row["candidate_id"]

        label = (
            1
            if candidate_id in gt_map.get(
                s1_id,
                set(),
            )
            else 0
        )

        labels.append(label)

    feats["label"] = labels

    positive_count = int(
        feats["label"].sum()
    )

    negative_count = int(
        len(feats) - positive_count
    )

    print(
        "Positive pairs:",
        positive_count,
    )

    print(
        "Negative pairs:",
        negative_count,
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    X = feats.drop(
        columns=[
            "source1_entity_id",
            "candidate_id",
            "label",
        ]
    )

    y = feats["label"]

    print("\nTraining matching model...")

    model = train_model(
        X,
        y,
        model_path=str(
            cfg.model_path
        ),
    )

    print("\nTraining complete.")
    print(
        "Model saved to:",
        cfg.model_path,
    )


if __name__ == "__main__":
    main()