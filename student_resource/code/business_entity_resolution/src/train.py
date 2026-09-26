from __future__ import annotations

import json
import logging
import random
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = Path(__file__).resolve().parent

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

TRAIN_DIR = DATA_ROOT / "dataset" / "train"
MODEL_DIR = REPO_ROOT / "student_resource" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "matcher.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"

TRAIN_SAMPLE_S1 = 5000
S2_SAMPLE_SIZE = 10000
S3_SAMPLE_SIZE = 10000
NEGATIVES_PER_S1 = 20

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

POSITIVE_WEIGHT = 8.0
NEGATIVE_WEIGHT = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Dynamic module loader
# ============================================================

def load_module_from_path(
    module_name: str,
    file_path: Path,
    package_dir: Path | None = None,
):
    """
    Load a Python module from a file path.

    package_dir is added to sys.path so that package-relative
    dependencies can be imported normally.
    """

    if package_dir is not None:
        package_dir = package_dir.resolve()
        package_parent = package_dir.parent

        if str(package_parent) not in sys.path:
            sys.path.insert(0, str(package_parent))

    spec = spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = module_from_spec(spec)

    # Register module before execution.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================
# Load Member 1 blocking module
# ============================================================

BLOCKING_PATH = REPO_ROOT / "src" / "blocking" / "candidate_generation.py"

blocking_module = load_module_from_path(
    "member1_blocking_for_training",
    BLOCKING_PATH,
    REPO_ROOT / "src",
)

generate_candidate_pairs = blocking_module.generate_candidates


# ============================================================
# Load Member 2 feature engineering
# ============================================================

FEATURE_PATH = SRC_DIR / "feature_engineering.py"

# IMPORTANT:
# feature_engineering.py uses:
#
# from .normalization import normalize_name, normalize_address
#
# Therefore it must be loaded as part of a package.

PACKAGE_NAME = "business_entity_resolution_training"

package_module = type(sys)(PACKAGE_NAME)
package_module.__path__ = [str(SRC_DIR)]
sys.modules[PACKAGE_NAME] = package_module

feature_module = load_module_from_path(
    f"{PACKAGE_NAME}.feature_engineering",
    FEATURE_PATH,
    SRC_DIR,
)

build_feature_matrix = feature_module.build_feature_matrix


# ============================================================
# Data loading
# ============================================================

def load_raw_source(path: Path) -> pd.DataFrame:
    logger.info("Loading %s", path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    return df.fillna("")


def load_ground_truth(path: Path) -> pd.DataFrame:
    logger.info("Loading ground truth: %s", path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")

    return df


# ============================================================
# Sampling
# ============================================================

def sample_source(
    path: Path,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:

    df = load_raw_source(path)

    if len(df) <= sample_size:
        return df

    return df.sample(
        n=sample_size,
        random_state=seed,
    ).reset_index(drop=True)


# ============================================================
# Ground-truth positive extraction
# ============================================================

def parse_ground_truth_links(
    ground_truth: pd.DataFrame,
    selected_s1_ids: set[str],
) -> dict[str, list[str]]:

    mapping: dict[str, list[str]] = {}

    for _, row in ground_truth.iterrows():

        s1_id = str(row["source1_entity_id"])

        if s1_id not in selected_s1_ids:
            continue

        raw_matches = str(row["matched_entity_ids"]).strip()

        if not raw_matches:
            mapping[s1_id] = []
            continue

        ids = [
            x.strip()
            for x in raw_matches.split(",")
            if x.strip()
        ]

        mapping[s1_id] = ids

    return mapping


# ============================================================
# Retrieve required positive records from full source files
# ============================================================

def lookup_records_by_ids(
    path: Path,
    required_ids: set[str],
    chunksize: int = 100_000,
) -> pd.DataFrame:

    if not required_ids:
        return pd.DataFrame()

    logger.info(
        "Looking up %d required positive IDs in %s",
        len(required_ids),
        path.name,
    )

    found_parts = []

    remaining = set(required_ids)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):

        chunk = chunk.fillna("")

        matches = chunk[
            chunk["entity_id"].isin(remaining)
        ]

        if not matches.empty:
            found_parts.append(matches.copy())

            remaining.difference_update(
                matches["entity_id"].astype(str).tolist()
            )

        if not remaining:
            break

    if not found_parts:
        logger.warning(
            "None of the required IDs were found in %s",
            path.name,
        )
        return pd.DataFrame()

    result = pd.concat(
        found_parts,
        ignore_index=True,
    )

    logger.info(
        "Found %d/%d required IDs in %s",
        result["entity_id"].nunique(),
        len(required_ids),
        path.name,
    )

    return result


# ============================================================
# Build controlled candidate set
# ============================================================

def build_candidate_set(
    s1_df: pd.DataFrame,
    sampled_s2: pd.DataFrame,
    sampled_s3: pd.DataFrame,
    positive_links: dict[str, list[str]],
) -> pd.DataFrame:

    logger.info("Generating blocking candidates...")

    candidate_parts = []

    if not sampled_s2.empty:

        pairs_s2 = generate_candidate_pairs(
            s1_df,
            sampled_s2,
        )

        if not pairs_s2.empty:
            pairs_s2 = pairs_s2.copy()
            pairs_s2["source"] = "source2"
            candidate_parts.append(pairs_s2)

    if not sampled_s3.empty:

        pairs_s3 = generate_candidate_pairs(
            s1_df,
            sampled_s3,
        )

        if not pairs_s3.empty:
            pairs_s3 = pairs_s3.copy()
            pairs_s3["source"] = "source3"
            candidate_parts.append(pairs_s3)

    if candidate_parts:
        candidates = pd.concat(
            candidate_parts,
            ignore_index=True,
        )
    else:
        candidates = pd.DataFrame(
            columns=[
                "source1_entity_id",
                "matched_entity_id",
                "source",
            ]
        )

    # --------------------------------------------------------
    # Add ALL ground-truth positives for selected S1 entities.
    #
    # This ensures that matcher training does not suffer from
    # positive-class starvation merely because a sampled source
    # happened not to contain a positive record.
    # --------------------------------------------------------

    positive_rows = []

    for s1_id, matched_ids in positive_links.items():

        for matched_id in matched_ids:

            if str(matched_id).startswith("S2-"):
                source = "source2"
            elif str(matched_id).startswith("S3-"):
                source = "source3"
            else:
                continue

            positive_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "matched_entity_id": matched_id,
                    "source": source,
                }
            )

    if positive_rows:

        positives_df = pd.DataFrame(positive_rows)

        candidates = pd.concat(
            [
                candidates,
                positives_df,
            ],
            ignore_index=True,
        )

    candidates = candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "matched_entity_id",
        ]
    ).reset_index(drop=True)

    logger.info(
        "Controlled candidate pairs: %s",
        f"{len(candidates):,}",
    )

    return candidates


# ============================================================
# Restrict source data to candidates
# ============================================================

def prepare_feature_sources(
    sampled_source: pd.DataFrame,
    positive_source: pd.DataFrame,
) -> pd.DataFrame:

    frames = []

    if sampled_source is not None and not sampled_source.empty:
        frames.append(sampled_source)

    if positive_source is not None and not positive_source.empty:
        frames.append(positive_source)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result = result.drop_duplicates(
        subset=["entity_id"]
    ).reset_index(drop=True)

    return result


# ============================================================
# Labels
# ============================================================

def build_labels(
    candidates: pd.DataFrame,
    positive_links: dict[str, list[str]],
) -> np.ndarray:

    positive_pairs = set()

    for s1_id, matched_ids in positive_links.items():
        for matched_id in matched_ids:
            positive_pairs.add(
                (
                    str(s1_id),
                    str(matched_id),
                )
            )

    labels = []

    for _, row in candidates.iterrows():

        pair = (
            str(row["source1_entity_id"]),
            str(row["matched_entity_id"]),
        )

        labels.append(
            1 if pair in positive_pairs else 0
        )

    return np.asarray(labels, dtype=np.int8)


# ============================================================
# Negative downsampling
# ============================================================

def reduce_negatives(
    candidates: pd.DataFrame,
    labels: np.ndarray,
    negatives_per_s1: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:

    rng = np.random.default_rng(seed)

    candidates = candidates.copy()
    candidates["_label"] = labels

    selected_parts = []

    positive_df = candidates[
        candidates["_label"] == 1
    ]

    if not positive_df.empty:
        selected_parts.append(positive_df)

    negative_df = candidates[
        candidates["_label"] == 0
    ]

    if not negative_df.empty:

        for s1_id, group in negative_df.groupby(
            "source1_entity_id"
        ):

            if len(group) <= negatives_per_s1:
                selected_parts.append(group)
                continue

            indices = rng.choice(
                len(group),
                size=negatives_per_s1,
                replace=False,
            )

            selected_parts.append(
                group.iloc[indices]
            )

    if not selected_parts:
        return (
            pd.DataFrame(
                columns=candidates.columns
            ),
            np.asarray([], dtype=np.int8),
        )

    result = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    result = result.drop(
        columns=["_label"]
    )

    result_labels = build_labels_from_positive_set(
        result,
        positive_df,
    )

    return result, result_labels


def build_labels_from_positive_set(
    candidates: pd.DataFrame,
    positive_df: pd.DataFrame,
) -> np.ndarray:

    positive_pairs = set(
        zip(
            positive_df["source1_entity_id"].astype(str),
            positive_df["matched_entity_id"].astype(str),
        )
    )

    labels = []

    for _, row in candidates.iterrows():

        pair = (
            str(row["source1_entity_id"]),
            str(row["matched_entity_id"]),
        )

        labels.append(
            1 if pair in positive_pairs else 0
        )

    return np.asarray(labels, dtype=np.int8)


# ============================================================
# Entity-level train/validation split
# ============================================================

def split_by_entity(
    candidates: pd.DataFrame,
    s1_ids: list[str],
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_ids, validation_ids = train_test_split(
        s1_ids,
        test_size=validation_fraction,
        random_state=seed,
    )

    train_set = set(train_ids)
    validation_set = set(validation_ids)

    train_candidates = candidates[
        candidates["source1_entity_id"].isin(train_set)
    ].reset_index(drop=True)

    validation_candidates = candidates[
        candidates["source1_entity_id"].isin(validation_set)
    ].reset_index(drop=True)

    logger.info(
        "Training S1 entities: %s",
        f"{len(train_set):,}",
    )

    logger.info(
        "Validation S1 entities: %s",
        f"{len(validation_set):,}",
    )

    return train_candidates, validation_candidates


# ============================================================
# Entity-level evaluation
# ============================================================

def evaluate_entity_level(
    validation_candidates: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    positive_links: dict[str, list[str]],
):

    validation_candidates = validation_candidates.copy()

    validation_candidates["probability"] = probabilities

    predicted_pairs = validation_candidates[
        validation_candidates["probability"] >= threshold
    ]

    macro_precision = []
    macro_recall = []
    macro_f05 = []

    true_singletons = 0
    predicted_singletons = 0

    beta = 0.5

    for s1_id, group in validation_candidates.groupby(
        "source1_entity_id"
    ):

        true_ids = set(
            positive_links.get(
                str(s1_id),
                [],
            )
        )

        predicted_ids = set(
            predicted_pairs.loc[
                predicted_pairs["source1_entity_id"]
                == s1_id,
                "matched_entity_id",
            ].astype(str)
        )

        if len(true_ids) == 1:
            true_singletons += 1

        if len(predicted_ids) == 1:
            predicted_singletons += 1

        if not true_ids and not predicted_ids:
            precision = 1.0
            recall = 1.0
        elif not true_ids and predicted_ids:
            precision = 0.0
            recall = 0.0
        elif true_ids and not predicted_ids:
            precision = 1.0
            recall = 0.0
        else:
            tp = len(true_ids & predicted_ids)
            fp = len(predicted_ids - true_ids)
            fn = len(true_ids - predicted_ids)

            precision = (
                tp / (tp + fp)
                if (tp + fp)
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn)
                else 0.0
            )

        if precision + recall == 0:
            f05 = 0.0
        else:
            f05 = (
                (1 + beta**2)
                * precision
                * recall
                / (
                    beta**2 * precision
                    + recall
                )
            )

        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f05.append(f05)

    return {
        "macro_precision": float(
            np.mean(macro_precision)
        ),
        "macro_recall": float(
            np.mean(macro_recall)
        ),
        "macro_f05": float(
            np.mean(macro_f05)
        ),
        "true_singletons": true_singletons,
        "predicted_singletons": predicted_singletons,
    }


# ============================================================
# Main training
# ============================================================

if __name__ == "__main__":

    logger.info("Starting controlled matcher training")

    logger.info(
        "DATA_ROOT: %s",
        DATA_ROOT,
    )

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    s1_path = TRAIN_DIR / "train_source1.tsv"

    s1 = load_raw_source(s1_path)

    if len(s1) > TRAIN_SAMPLE_S1:
        s1 = s1.sample(
            n=TRAIN_SAMPLE_S1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

    logger.info(
        "S1 rows loaded: %s",
        f"{len(s1):,}",
    )

    selected_s1_ids = set(
        s1["entity_id"].astype(str)
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    positive_links = parse_ground_truth_links(
        ground_truth,
        selected_s1_ids,
    )

    total_true_links = sum(
        len(v)
        for v in positive_links.values()
    )

    logger.info(
        "Ground-truth S1 rows selected: %s",
        f"{len(positive_links):,}",
    )

    logger.info(
        "Ground-truth positive links: %s",
        f"{total_true_links:,}",
    )

    # --------------------------------------------------------
    # Sample sources for negative candidate generation
    # --------------------------------------------------------

    s2_sample = sample_source(
        TRAIN_DIR / "train_source2.tsv",
        S2_SAMPLE_SIZE,
        RANDOM_SEED,
    )

    s3_sample = sample_source(
        TRAIN_DIR / "train_source3.tsv",
        S3_SAMPLE_SIZE,
        RANDOM_SEED + 1,
    )

    logger.info(
        "S2 sample rows: %s",
        f"{len(s2_sample):,}",
    )

    logger.info(
        "S3 sample rows: %s",
        f"{len(s3_sample):,}",
    )

    # --------------------------------------------------------
    # Retrieve ALL positive source rows from full files
    # --------------------------------------------------------

    required_s2_ids = set()
    required_s3_ids = set()

    for matched_ids in positive_links.values():

        for matched_id in matched_ids:

            matched_id = str(matched_id)

            if matched_id.startswith("S2-"):
                required_s2_ids.add(matched_id)

            elif matched_id.startswith("S3-"):
                required_s3_ids.add(matched_id)

    logger.info(
        "Required S2 positive IDs: %s",
        f"{len(required_s2_ids):,}",
    )

    logger.info(
        "Required S3 positive IDs: %s",
        f"{len(required_s3_ids):,}",
    )

    s2_positive = lookup_records_by_ids(
        TRAIN_DIR / "train_source2.tsv",
        required_s2_ids,
    )

    s3_positive = lookup_records_by_ids(
        TRAIN_DIR / "train_source3.tsv",
        required_s3_ids,
    )

    # --------------------------------------------------------
    # Build candidate set
    # --------------------------------------------------------

    candidates = build_candidate_set(
        s1,
        s2_sample,
        s3_sample,
        positive_links,
    )

    # --------------------------------------------------------
    # Labels before negative reduction
    # --------------------------------------------------------

    labels_before = build_labels(
        candidates,
        positive_links,
    )

    logger.info(
        "Candidate positives before negative sampling: %s",
        f"{int(labels_before.sum()):,}",
    )

    logger.info(
        "Candidate negatives before negative sampling: %s",
        f"{int((labels_before == 0).sum()):,}",
    )

    # --------------------------------------------------------
    # Negative sampling
    # --------------------------------------------------------

    candidates, labels = reduce_negatives(
        candidates,
        labels_before,
        NEGATIVES_PER_S1,
        RANDOM_SEED,
    )

    logger.info(
        "Candidates after negative sampling: %s",
        f"{len(candidates):,}",
    )

    logger.info(
        "Positive examples: %s",
        f"{int(labels.sum()):,}",
    )

    logger.info(
        "Negative examples: %s",
        f"{int((labels == 0).sum()):,}",
    )

    # --------------------------------------------------------
    # Entity-level split
    # --------------------------------------------------------

    selected_ids = sorted(
        candidates["source1_entity_id"]
        .astype(str)
        .unique()
    )

    train_candidates, validation_candidates = split_by_entity(
        candidates,
        selected_ids,
        VALIDATION_FRACTION,
        RANDOM_SEED,
    )

    train_labels = build_labels(
        train_candidates,
        positive_links,
    )

    validation_labels = build_labels(
        validation_candidates,
        positive_links,
    )

    logger.info(
        "Training positives: %s",
        f"{int(train_labels.sum()):,}",
    )

    logger.info(
        "Training negatives: %s",
        f"{int((train_labels == 0).sum()):,}",
    )

    logger.info(
        "Validation positives: %s",
        f"{int(validation_labels.sum()):,}",
    )

    logger.info(
        "Validation negatives: %s",
        f"{int((validation_labels == 0).sum()):,}",
    )

    # --------------------------------------------------------
    # Prepare feature source tables
    # --------------------------------------------------------

    feature_s2 = prepare_feature_sources(
        s2_sample,
        s2_positive,
    )

    feature_s3 = prepare_feature_sources(
        s3_sample,
        s3_positive,
    )

    logger.info(
        "Feature S2 rows: %s",
        f"{len(feature_s2):,}",
    )

    logger.info(
        "Feature S3 rows: %s",
        f"{len(feature_s3):,}",
    )

    # --------------------------------------------------------
    # Generate training features
    # --------------------------------------------------------

    logger.info(
        "Generating training features..."
    )

    X_train = build_feature_matrix(
        train_candidates,
        s1,
        feature_s2,
        feature_s3,
    )

    logger.info(
        "Training feature matrix shape: %s",
        X_train.shape,
    )

    logger.info(
        "Generating validation features..."
    )

    X_validation = build_feature_matrix(
        validation_candidates,
        s1,
        feature_s2,
        feature_s3,
    )

    logger.info(
        "Validation feature matrix shape: %s",
        X_validation.shape,
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    logger.info("Training HistGradientBoostingClassifier...")

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    sample_weights = np.where(
        train_labels == 1,
        POSITIVE_WEIGHT,
        NEGATIVE_WEIGHT,
    )

    model.fit(
        X_train,
        train_labels,
        sample_weight=sample_weights,
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Threshold tuning
    # --------------------------------------------------------

    best_threshold = None
    best_metrics = None

    logger.info(
        "Tuning threshold..."
    )

    for threshold in np.arange(
        0.01,
        0.91,
        0.01,
    ):

        metrics = evaluate_entity_level(
            validation_candidates,
            validation_probabilities,
            float(threshold),
            positive_links,
        )

        if (
            best_metrics is None
            or metrics["macro_f05"]
            > best_metrics["macro_f05"]
        ):
            best_metrics = metrics
            best_threshold = float(
                round(threshold, 2)
            )

    logger.info(
        "Best validation threshold: %.2f",
        best_threshold,
    )

    logger.info(
        "Validation Macro Precision: %.4f",
        best_metrics["macro_precision"],
    )

    logger.info(
        "Validation Macro Recall: %.4f",
        best_metrics["macro_recall"],
    )

    logger.info(
        "Validation Macro F0.5: %.4f",
        best_metrics["macro_f05"],
    )

    logger.info(
        "True singleton entities: %s",
        f"{best_metrics['true_singletons']:,}",
    )

    logger.info(
        "Predicted singleton entities: %s",
        f"{best_metrics['predicted_singletons']:,}",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        model,
        MODEL_PATH,
    )

    logger.info(
        "Model saved to: %s",
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    feature_names = list(
        X_train.columns
        if hasattr(X_train, "columns")
        else []
    )

    config = {
        "training_type": "controlled_positive_retrieval",
        "s1_sample_size": TRAIN_SAMPLE_S1,
        "s2_sample_size": S2_SAMPLE_SIZE,
        "s3_sample_size": S3_SAMPLE_SIZE,
        "negative_candidates_per_s1": NEGATIVES_PER_S1,
        "validation_fraction": VALIDATION_FRACTION,
        "random_seed": RANDOM_SEED,
        "positive_weight": POSITIVE_WEIGHT,
        "negative_weight": NEGATIVE_WEIGHT,
        "threshold": best_threshold,
        "validation_macro_precision": best_metrics[
            "macro_precision"
        ],
        "validation_macro_recall": best_metrics[
            "macro_recall"
        ],
        "validation_f05": best_metrics[
            "macro_f05"
        ],
        "feature_names": feature_names,
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    logger.info(
        "Config saved to: %s",
        CONFIG_PATH,
    )

    logger.info(
        "Controlled matcher training completed."
    )