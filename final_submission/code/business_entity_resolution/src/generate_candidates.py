from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import logging
import os
import sys
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "student_resource"

DATA_ROOT = Path(
    os.environ.get(
        "AMON_DATA_ROOT",
        str(DEFAULT_DATA_ROOT),
    )
).resolve()

MEMBER1_BLOCKING_FILE = (
    PACKAGE_ROOT
    / "src"
    / "blocking"
    / "candidate_generation.py"
)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Member 1 import
# ----------------------------------------------------------------------

import types


def load_member1_blocking_module():
    """
    Load Member 1's blocking implementation directly from its file.

    This avoids conflicts between:
        student_resource/code/business_entity_resolution/src
    and:
        repository/src
    """

    member1_src = PACKAGE_ROOT / "src"
    member1_blocking = member1_src / "blocking"
    member1_preprocessing = member1_src / "preprocessing"
    blocking_file = member1_blocking / "candidate_generation.py"

    if not blocking_file.exists():
        raise FileNotFoundError(
            f"Member 1 blocking file not found:\n{blocking_file}"
        )

    # 1. Load Member 1 normalize module
    norm_spec = importlib.util.spec_from_file_location(
        "member1_normalize",
        member1_preprocessing / "normalize.py",
    )
    if norm_spec is None or norm_spec.loader is None:
        raise ImportError("Could not load Member 1 normalize.py")
    norm_mod = importlib.util.module_from_spec(norm_spec)
    norm_spec.loader.exec_module(norm_mod)

    # 2. Register src.preprocessing compatibility in sys.modules
    if "src.preprocessing" not in sys.modules:
        pkg = types.ModuleType("src.preprocessing")
        pkg.__path__ = [str(member1_preprocessing)]
        sys.modules["src.preprocessing"] = pkg
    sys.modules["src.preprocessing.normalize"] = norm_mod

    # 3. Load candidate generation module
    spec = importlib.util.spec_from_file_location(
        "member1_candidate_generation",
        blocking_file,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load Member 1 blocker.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["member1_candidate_generation"] = module
    spec.loader.exec_module(module)

    return module


member1 = load_member1_blocking_module()

generate_candidates = member1.generate_candidates
load_source = member1.load_source


# ----------------------------------------------------------------------
# Data paths
# ----------------------------------------------------------------------

def get_dataset_dir(mode: str) -> Path:
    if mode == "test":
        return DATA_ROOT / "dataset" / "test"

    if mode == "train":
        return DATA_ROOT / "dataset" / "train"

    raise ValueError(f"Unsupported mode: {mode}")


def get_source_paths(mode: str):
    dataset_dir = get_dataset_dir(mode)

    if mode == "test":
        names = (
            "test_source1.tsv",
            "test_source2.tsv",
            "test_source3.tsv",
        )
    else:
        names = (
            "train_source1.tsv",
            "train_source2.tsv",
            "train_source3.tsv",
        )

    paths = tuple(dataset_dir / name for name in names)

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found:\n{path}"
            )

    return paths


def _write_one_source_candidates(source1_df, source_path: Path, temp_path: Path, source_label: str):
    """Generate one source's bounded candidate set without materializing pairs."""
    source_df = load_source(source_path)
    log.info("Building %s bounded blocking index (%s rows)...", source_label, f"{len(source_df):,}")
    source_index = member1.build_capped_source_index(source_df, max_block_size=20)
    del source_df
    gc.collect()

    pair_count = 0
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, country, name, address in source1_df[
            ["entity_id", "country_norm", "name_norm", "address_norm"]
        ].itertuples(index=False, name=None):
            candidate_ids = member1.candidate_ids_for_record(
                country, name, address, source_index
            )
            pair_count += len(candidate_ids)
            writer.writerow([s1_id, ",".join(sorted(candidate_ids))])

    del source_index
    gc.collect()
    log.info(f"{source_label} candidates: {pair_count:,}")
    return pair_count


def _run_streaming_blocking(mode: str, output_path: str | Path | None = None) -> pd.DataFrame:
    s1_path, s2_path, s3_path = get_source_paths(mode)
    source1_df = load_source(s1_path)
    log.info("S1 entities: %s", f"{len(source1_df):,}")
    if output_path is None:
        output_path = DATA_ROOT / "output" / "candidate_pairs.tsv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_s2 = output_path.with_name(output_path.name + ".source2.tmp.tsv")
    temp_s3 = output_path.with_name(output_path.name + ".source3.tmp.tsv")
    temp_final = output_path.with_name(output_path.name + ".tmp")

    _write_one_source_candidates(source1_df, s2_path, temp_s2, "Source 2")
    _write_one_source_candidates(source1_df, s3_path, temp_s3, "Source 3")

    rows = []
    with temp_s2.open("r", encoding="utf-8", newline="") as f2, \
         temp_s3.open("r", encoding="utf-8", newline="") as f3, \
         temp_final.open("w", encoding="utf-8", newline="") as fout:
        r2, r3 = csv.reader(f2, delimiter="\t"), csv.reader(f3, delimiter="\t")
        w = csv.writer(fout, delimiter="\t", lineterminator="\n")
        header2, header3 = next(r2), next(r3)
        expected = ["source1_entity_id", "candidate_entity_ids"]
        if header2 != expected or header3 != expected:
            raise ValueError("Unexpected temporary candidate schema")
        w.writerow(expected)
        for row2, row3 in zip(r2, r3, strict=True):
            if row2[0] != row3[0]:
                raise ValueError("Source candidate passes have different S1 ordering")
            ids = set(filter(None, row2[1].split(",")))
            ids.update(filter(None, row3[1].split(",")))
            w.writerow([row2[0], ",".join(sorted(ids))])
            if len(rows) < 5:
                rows.append([row2[0], ",".join(sorted(ids))])

    temp_final.replace(output_path)
    temp_s2.unlink()
    temp_s3.unlink()
    log.info("Wrote complete candidate file: %s", output_path)
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])


# ----------------------------------------------------------------------
# Candidate generation
# ----------------------------------------------------------------------

def run_member1_blocking(
    test_or_train: str = "test",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run Member 1 blocking for the selected dataset split.

    IMPORTANT:
    This performs full candidate generation for the selected split.
    The caller must explicitly request it.
    """

    return _run_streaming_blocking(test_or_train, output_path)

    dataset_dir = get_dataset_dir(test_or_train)

    s1_path, s2_path, s3_path = get_source_paths(test_or_train)

    log.info("=" * 70)
    log.info("AMON candidate generation")
    log.info("=" * 70)
    log.info("Mode: %s", test_or_train)
    log.info("DATA_ROOT: %s", DATA_ROOT)
    log.info("Dataset directory: %s", dataset_dir)

    log.info("Loading Source 1...")
    source1_df = load_source(s1_path)

    log.info("Loading Source 2...")
    source2_df = load_source(s2_path)

    log.info("Loading Source 3...")
    source3_df = load_source(s3_path)

    log.info(
        "Rows: S1=%s | S2=%s | S3=%s",
        f"{len(source1_df):,}",
        f"{len(source2_df):,}",
        f"{len(source3_df):,}",
    )

    log.info("")
    log.info("Generating S1 -> S2 candidates...")

    candidates_s2 = generate_candidates(
        source1_df,
        source2_df,
    )

    candidates_s2 = candidates_s2.rename(
        columns={
            "matched_entity_id": "candidate_id",
        }
    )

    candidates_s2["source"] = "source2"

    log.info(
        "S1 -> S2 candidate pairs: %,d",
        len(candidates_s2),
    )

    log.info("")
    log.info("Generating S1 -> S3 candidates...")

    candidates_s3 = generate_candidates(
        source1_df,
        source3_df,
    )

    candidates_s3 = candidates_s3.rename(
        columns={
            "matched_entity_id": "candidate_id",
        }
    )

    candidates_s3["source"] = "source3"

    log.info(
        "S1 -> S3 candidate pairs: %,d",
        len(candidates_s3),
    )

    # Convert both source candidate sets to the challenge schema.
    candidates_s2 = candidates_s2[
        [
            "source1_entity_id",
            "candidate_id",
        ]
    ]

    candidates_s3 = candidates_s3[
        [
            "source1_entity_id",
            "candidate_id",
        ]
    ]

    combined = pd.concat(
        [
            candidates_s2,
            candidates_s3,
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_id",
        ]
    )

    grouped = (
        combined
        .groupby("source1_entity_id")["candidate_id"]
        .apply(
            lambda values: ",".join(
                sorted(
                    set(
                        str(value)
                        for value in values
                        if pd.notna(value)
                    )
                )
            )
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "candidate_id": "candidate_entity_ids"
        }
    )

    # Every S1 entity must appear in the final candidate file.
    all_s1 = source1_df[
        ["entity_id"]
    ].rename(
        columns={
            "entity_id": "source1_entity_id"
        }
    )

    grouped = all_s1.merge(
        grouped,
        on="source1_entity_id",
        how="left",
    )

    grouped["candidate_entity_ids"] = (
        grouped["candidate_entity_ids"]
        .fillna("")
    )

    grouped["candidate_entity_ids"] = (
        grouped["candidate_entity_ids"]
        .astype(str)
    )

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        log.info("")
        log.info(
            "Writing candidates to: %s",
            output_path,
        )

        grouped.to_csv(
            output_path,
            sep="\t",
            index=False,
        )

        log.info(
            "Wrote %,d S1 rows",
            len(grouped),
        )

    log.info("")
    log.info("=" * 70)
    log.info("Candidate generation completed")
    log.info("=" * 70)

    return grouped


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate AMON business entity candidate pairs using "
            "Member 1 blocking."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["test", "train"],
        required=True,
        help=(
            "Dataset split to process. "
            "This performs full candidate generation."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output TSV path. "
            "Defaults to student_resource/output/candidate_pairs.tsv."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_file = args.output

    if output_file is None:
        output_file = (
            DATA_ROOT
            / "output"
            / "candidate_pairs.tsv"
        )

    candidates = run_member1_blocking(
        test_or_train=args.mode,
        output_path=output_file,
    )

    print("")
    print("=" * 70)
    print("SAMPLE CANDIDATES")
    print("=" * 70)

    print(
        candidates.head(5).to_string(
            index=False
        )
    )

    print("")


if __name__ == "__main__":
    main()
