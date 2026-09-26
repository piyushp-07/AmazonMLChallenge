from dataclasses import dataclass
from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[4]

_default_data_root = REPO_ROOT / "student_resource"
if not (_default_data_root / "dataset").exists():
    _downloads_data_root = Path(
        r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
    )
    if (_downloads_data_root / "dataset").exists():
        _default_data_root = _downloads_data_root

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
    model_path: Path = REPO_ROOT / "student_resource" / "models" / "matcher.pkl"


cfg = Config()