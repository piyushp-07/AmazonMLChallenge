import pandas as pd
from typing import Tuple
from .config import cfg


def load_train_sources(train_dir=cfg.train_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_csv(str(train_dir / "train_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(str(train_dir / "train_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(str(train_dir / "train_source3.tsv"), sep="\t", dtype=str)
    return s1, s2, s3


def load_train_ground_truth(train_dir=cfg.train_dir) -> pd.DataFrame:
    return pd.read_csv(str(train_dir / "train_ground_truth.tsv"), sep="\t", dtype=str)


def load_test_sources(test_dir=cfg.test_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1 = pd.read_csv(str(test_dir / "test_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(str(test_dir / "test_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(str(test_dir / "test_source3.tsv"), sep="\t", dtype=str)
    return s1, s2, s3


def load_candidate_pairs(path=cfg.candidate_path) -> pd.DataFrame:
    return pd.read_csv(str(path), sep="\t", dtype=str)
