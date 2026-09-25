import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

from .config import cfg
from .feature_engineering import features_from_pairs
from .training import train_model

# Member 1 repo is a sibling to the extracted dataset under the workspace root.
MEMBER1_ROOT = Path(__file__).resolve().parents[4] / "member1_repo"

# The Member 1 repo and the Member 2 code both use the top-level package name `src`.
# To avoid shadowing, we load the Member 1 blocking modules directly from disk under a
# synthetic package tree rooted at the Member 1 repo.
member1_src = MEMBER1_ROOT / "src"
member1_blocking = member1_src / "blocking"
member1_preprocessing = member1_src / "preprocessing"

for name, path in {
    "src": member1_src,
    "src.blocking": member1_blocking,
    "src.preprocessing": member1_preprocessing,
}.items():
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


member1_normalize = _load_module(
    "src.preprocessing.normalize",
    member1_preprocessing / "normalize.py",
)
member1_blocking_module = _load_module(
    "src.blocking.candidate_generation",
    member1_blocking / "candidate_generation.py",
)

generate_candidates = member1_blocking_module.generate_candidates
load_source = member1_blocking_module.load_source


def _combine_candidates(source1_df, source2_df, source3_df):
    s2_pairs = generate_candidates(source1_df, source2_df).rename(columns={"matched_entity_id": "candidate_entity_id"})
    s3_pairs = generate_candidates(source1_df, source3_df).rename(columns={"matched_entity_id": "candidate_entity_id"})
    combined = pd.concat([s2_pairs, s3_pairs], ignore_index=True)
    combined = combined.groupby("source1_entity_id")["candidate_entity_id"].agg(lambda ids: ",".join(sorted(set(ids))))
    combined = combined.reset_index().rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    return combined


def generate_candidate_pairs(train_or_test: str = "train"):
    root = cfg.root_dir
    if train_or_test == "train":
        s1 = load_source(str(cfg.train_dir / "train_source1.tsv"))
        s2 = load_source(str(cfg.train_dir / "train_source2.tsv"))
        s3 = load_source(str(cfg.train_dir / "train_source3.tsv"))
    else:
        s1 = load_source(str(cfg.test_dir / "test_source1.tsv"))
        s2 = load_source(str(cfg.test_dir / "test_source2.tsv"))
        s3 = load_source(str(cfg.test_dir / "test_source3.tsv"))

    candidates = _combine_candidates(s1, s2, s3)
    candidates["source1_entity_id"] = candidates["source1_entity_id"].astype(str)
    candidates = candidates[["source1_entity_id", "candidate_entity_ids"]]
    return candidates


def build_training_features(train_candidates: pd.DataFrame):
    s1, s2, s3 = __import__("src.data_loader", fromlist=["load_train_sources"]).load_train_sources()
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    return features_from_pairs(s1, train_candidates, s2s3)


def main():
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)

    train_candidates = generate_candidate_pairs("train")
    train_candidates.to_csv(str(cfg.output_dir / "candidate_pairs.tsv"), sep="\t", index=False)

    gt = __import__("src.data_loader", fromlist=["load_train_ground_truth"]).load_train_ground_truth()
    gt_map = {}
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        val = row["matched_entity_ids"]
        gt_map[s1_id] = set([v.strip() for v in (val.split(",") if val else []) if v.strip()])

    feats = build_training_features(train_candidates)
    if feats.empty:
        print("No training candidates found.")
        return

    labels = []
    for _, row in feats.iterrows():
        s1_id = row["source1_entity_id"]
        cid = row["candidate_id"]
        labels.append(1 if cid in gt_map.get(s1_id, set()) else 0)
    feats["label"] = labels

    X = feats.drop(columns=["source1_entity_id", "candidate_id", "label"])
    y = feats["label"]
    model = train_model(X, y, model_path=str(cfg.model_path))
    print("Training complete. Model saved to", cfg.model_path)
    print("Candidate pairs generated:", len(train_candidates))


if __name__ == "__main__":
    main()
