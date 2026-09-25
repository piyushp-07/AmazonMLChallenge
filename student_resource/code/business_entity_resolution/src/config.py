from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Config:
    seed: int = 42
    root_dir: Path = ROOT
    train_dir: Path = ROOT / "dataset" / "train"
    test_dir: Path = ROOT / "dataset" / "test"
    output_dir: Path = ROOT / "output"
    candidate_path: Path = ROOT / "output" / "candidate_pairs.tsv"
    matching_out: Path = ROOT / "output" / "matching_results.tsv"
    model_path: Path = ROOT / "models" / "matcher.pkl"


cfg = Config()
