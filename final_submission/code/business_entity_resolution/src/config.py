from dataclasses import dataclass
from pathlib import Path
import os


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]

_default_data_root = PACKAGE_ROOT

DATA_ROOT = Path(
    os.environ.get(
        "AMON_DATA_ROOT",
        str(_default_data_root),
    )
).resolve()


@dataclass
class Config:
    seed: int = 42
    root_dir: Path = DATA_ROOT
    train_dir: Path = DATA_ROOT / "dataset" / "train"
    test_dir: Path = DATA_ROOT / "dataset" / "test"
    output_dir: Path = DATA_ROOT / "output"
    candidate_path: Path = DATA_ROOT / "output" / "candidate_pairs.tsv"
    matching_out: Path = DATA_ROOT / "output" / "matching_results.tsv"
    model_path: Path = PACKAGE_ROOT / "models" / "matcher.pkl"


cfg = Config()
