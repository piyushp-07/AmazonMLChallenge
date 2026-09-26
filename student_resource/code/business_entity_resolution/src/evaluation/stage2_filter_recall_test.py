import sys
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline
from src.candidate_filter.cheap_filter import (
    build_candidate_filter_indexes,
    normalize_simple,
    name_tokens,
    address_tokens,
)


DATA_ROOT = Path(
    r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
)

TRAIN_DIR = DATA_ROOT / "dataset" / "train"


def numeric_parts(value):
    value = normalize_simple(value)

    if not value:
        return set()

    return set(
        re.findall(r"\d+", value)
    )


def keep_candidate_test(
    s1_row,
    candidate_row,
    indexes,
    rare_name_threshold=100,
):
    name1 = normalize_simple(
        s1_row.get("business_name", "")
    )

    name2 = normalize_simple(
        candidate_row.get("business_name", "")
    )

    address1 = normalize_simple(
        s1_row.get("business_address", "")
    )

    address2 = normalize_simple(
        candidate_row.get("business_address", "")
    )

    # ---------------------------------------------------------
    # Existing Stage-2 rules
    # ---------------------------------------------------------

    if (
        name1
        and name2
        and name1 == name2
    ):
        return True

    n1 = name_tokens(name1)
    n2 = name_tokens(name2)

    a1 = address_tokens(address1)
    a2 = address_tokens(address2)

    shared_name = n1 & n2
    shared_address = a1 & a2

    # Existing address rule
    if len(shared_address) >= 2:
        return True

    # Existing name + address rule
    if shared_name and shared_address:
        return True

    # Existing rare-name rule
    for token in shared_name:

        if (
            indexes["name_frequency"].get(
                token,
                0
            )
            <= rare_name_threshold
        ):
            return True

    # ---------------------------------------------------------
    # NEW TARGETED MULTILINGUAL ADDRESS RULE
    # ---------------------------------------------------------

    numbers1 = numeric_parts(address1)
    numbers2 = numeric_parts(address2)

    shared_numbers = numbers1 & numbers2

    if (
        shared_numbers
        and shared_address
    ):
        return True

    return False


def filter_candidates(
    s1_df,
    candidates_df,
    s2s3_df,
):
    indexes = build_candidate_filter_indexes(
        s2s3_df
    )

    source1_index = s1_df.set_index(
        "entity_id"
    )

    candidate_index = s2s3_df.set_index(
        "entity_id"
    )

    output_rows = []

    before_count = 0
    after_count = 0

    for _, row in candidates_df.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        if s1_id not in source1_index.index:
            continue

        s1_row = source1_index.loc[
            s1_id
        ]

        candidate_ids = [
            x.strip()
            for x in str(
                row.get(
                    "candidate_entity_ids",
                    ""
                )
            ).split(",")
            if x.strip()
        ]

        kept = []

        for candidate_id in candidate_ids:

            if (
                candidate_id
                not in candidate_index.index
            ):
                continue

            before_count += 1

            candidate_row = candidate_index.loc[
                candidate_id
            ]

            if keep_candidate_test(
                s1_row,
                candidate_row,
                indexes,
            ):
                kept.append(candidate_id)
                after_count += 1

        output_rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids":
                    ",".join(
                        sorted(
                            set(kept)
                        )
                    ),
            }
        )

    result = pd.DataFrame(
        output_rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids",
        ],
    )

    print(
        f"\nStage-2 candidates before: "
        f"{before_count:,}"
    )

    print(
        f"Stage-2 candidates after:  "
        f"{after_count:,}"
    )

    if before_count:

        print(
            "Retention:",
            f"{after_count / before_count * 100:.2f}%"
        )

    return result


def make_candidate_table(
    source1,
    source2,
    source_name,
):
    print(
        f"\nGenerating {source_name} candidates..."
    )

    candidates = pipeline.generate_candidates(
        source1,
        source2,
    )

    print(
        f"{source_name} candidate pairs: "
        f"{len(candidates):,}"
    )

    candidates = candidates.rename(
        columns={
            "matched_entity_id":
                "candidate_entity_id"
        }
    )

    return candidates


def candidate_map(candidates):

    result = {}

    for s1_id, group in candidates.groupby(
        "source1_entity_id"
    ):

        result[s1_id] = set(
            group[
                "candidate_entity_id"
            ].astype(str)
        )

    return result


def ground_truth_map(gt):

    result = {}

    for _, row in gt.iterrows():

        ids = {
            x.strip()
            for x in row[
                "matched_entity_ids"
            ].split(",")
            if x.strip()
        }

        result[
            row["source1_entity_id"]
        ] = ids

    return result


def recall_for_source(
    gt_map,
    candidates_map,
):

    total_true = 0
    recovered = 0

    for s1_id, true_ids in gt_map.items():

        total_true += len(true_ids)

        recovered += len(
            true_ids
            & candidates_map.get(
                s1_id,
                set()
            )
        )

    recall = (
        recovered / total_true * 100
        if total_true
        else 0
    )

    return (
        total_true,
        recovered,
        recall,
    )


def main():

    print("=" * 70)
    print(
        "STAGE-2 TARGETED RULE VALIDATION"
    )
    print(
        "S1 rows 5,000-9,999 + 10,000 S2 + 10,000 S3"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading S1...")

    s1 = pipeline.load_source(
        str(
            TRAIN_DIR
            / "train_source1.tsv"
        )
    ).iloc[10000:15000]

    print(
        f"S1 rows loaded: {len(s1):,}"
    )

    print("Loading S2...")

    s2 = pipeline.load_source(
        str(
            TRAIN_DIR
            / "train_source2.tsv"
        )
    ).head(10000)

    print(
        f"S2 rows loaded: {len(s2):,}"
    )

    print("Loading S3...")

    s3 = pipeline.load_source(
        str(
            TRAIN_DIR
            / "train_source3.tsv"
        )
    ).head(10000)

    print(
        f"S3 rows loaded: {len(s3):,}"
    )

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    gt = pd.read_csv(
        TRAIN_DIR
        / "train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1_ids = set(
        s1["entity_id"]
    )

    gt = gt[
        gt["source1_entity_id"].isin(
            s1_ids
        )
    ]

    gt_map = ground_truth_map(
        gt
    )

    # ---------------------------------------------------------
    # S2 candidates
    # ---------------------------------------------------------

    c2 = make_candidate_table(
        s1,
        s2,
        "S2",
    )

    s2_map = candidate_map(c2)

    s2_ids = set(
        s2["entity_id"]
    )

    gt_s2 = {
        s1_id:
            true_ids & s2_ids
        for s1_id, true_ids
        in gt_map.items()
    }

    (
        s2_true,
        s2_recovered,
        s2_recall,
    ) = recall_for_source(
        gt_s2,
        s2_map,
    )

    print(
        f"S2 true links:       {s2_true:,}"
    )

    print(
        f"S2 recovered:        "
        f"{s2_recovered:,}"
    )

    print(
        f"S2 blocking recall:  "
        f"{s2_recall:.4f}%"
    )

    # ---------------------------------------------------------
    # S3 candidates
    # ---------------------------------------------------------

    c3 = make_candidate_table(
        s1,
        s3,
        "S3",
    )

    s3_map = candidate_map(c3)

    s3_ids = set(
        s3["entity_id"]
    )

    gt_s3 = {
        s1_id:
            true_ids & s3_ids
        for s1_id, true_ids
        in gt_map.items()
    }

    (
        s3_true,
        s3_recovered,
        s3_recall,
    ) = recall_for_source(
        gt_s3,
        s3_map,
    )

    print(
        f"S3 true links:       {s3_true:,}"
    )

    print(
        f"S3 recovered:        "
        f"{s3_recovered:,}"
    )

    print(
        f"S3 blocking recall:  "
        f"{s3_recall:.4f}%"
    )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    combined = pd.concat(
        [c2, c3],
        ignore_index=True,
    )

    combined = (
        combined
        .groupby(
            "source1_entity_id"
        )[
            "candidate_entity_id"
        ]
        .agg(
            lambda x:
                ",".join(
                    sorted(
                        set(x)
                    )
                )
        )
        .reset_index()
        .rename(
            columns={
                "candidate_entity_id":
                    "candidate_entity_ids"
            }
        )
    )

    all_s1 = s1[
        ["entity_id"]
    ].rename(
        columns={
            "entity_id":
                "source1_entity_id"
        }
    )

    candidates = all_s1.merge(
        combined,
        on="source1_entity_id",
        how="left",
    )

    candidates[
        "candidate_entity_ids"
    ] = (
        candidates[
            "candidate_entity_ids"
        ].fillna("")
    )

    before_count = sum(
        len(
            [
                x
                for x in value.split(",")
                if x
            ]
        )
        for value in candidates[
            "candidate_entity_ids"
        ]
    )

    print(
        f"\nCombined Stage-1 candidates: "
        f"{before_count:,}"
    )

    # ---------------------------------------------------------
    # Stage 2
    # ---------------------------------------------------------

    print(
        "\nApplying Stage-2 "
        "with targeted numeric-address rule..."
    )

    source23 = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    filtered = filter_candidates(
        s1_df=s1,
        candidates_df=candidates,
        s2s3_df=source23,
    )

    after_count = sum(
        len(
            [
                x
                for x in value.split(",")
                if x
            ]
        )
        for value in filtered[
            "candidate_entity_ids"
        ]
    )

    # ---------------------------------------------------------
    # Filtered candidate map
    # ---------------------------------------------------------

    filtered_map = {}

    for _, row in filtered.iterrows():

        filtered_map[
            row["source1_entity_id"]
        ] = {
            x.strip()
            for x in row[
                "candidate_entity_ids"
            ].split(",")
            if x.strip()
        }

    # ---------------------------------------------------------
    # Recall
    # ---------------------------------------------------------

    all_true = 0
    all_before = 0
    all_after = 0

    removed_true_links = []

    for s1_id, true_ids in gt_map.items():

        validation_true = (
            true_ids
            & (s2_ids | s3_ids)
        )

        before = (
            validation_true
            & (
                s2_map.get(
                    s1_id,
                    set()
                )
                |
                s3_map.get(
                    s1_id,
                    set()
                )
            )
        )

        after = (
            validation_true
            & filtered_map.get(
                s1_id,
                set()
            )
        )

        all_true += len(
            validation_true
        )

        all_before += len(
            before
        )

        all_after += len(
            after
        )

        for candidate_id in (
            before - after
        ):

            if candidate_id in validation_true:

                removed_true_links.append(
                    (
                        s1_id,
                        candidate_id,
                    )
                )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TARGETED RULE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"True links in validation sample: "
        f"{all_true:,}"
    )

    print(
        f"Recovered before Stage-2: "
        f"{all_before:,}"
    )

    print(
        f"Recovered after Stage-2:  "
        f"{all_after:,}"
    )

    print(
        f"Stage-1 recall: "
        f"{all_before / all_true * 100:.4f}%"
    )

    print(
        f"Stage-2 recall: "
        f"{all_after / all_true * 100:.4f}%"
    )

    print(
        f"True links removed by Stage-2: "
        f"{len(removed_true_links):,}"
    )

    print(
        f"Candidate reduction: "
        f"{(1 - after_count / before_count) * 100:.2f}%"
    )

    print(
        "\nFirst 20 true links removed:"
    )

    for (
        s1_id,
        candidate_id,
    ) in removed_true_links[:20]:

        print(
            f"  {s1_id} -> "
            f"{candidate_id}"
        )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()