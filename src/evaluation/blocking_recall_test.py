import os
import pandas as pd

from src.blocking.candidate_generation import generate_candidates
from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"

TRAIN_DIR = os.path.join(DATASET_ROOT, "dataset", "train")

S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")


def prepare_source(df):
    df = df.copy()

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["address_norm"] = df["business_address"].map(
        normalize_address
    )

    df["country_norm"] = df["country"].map(
        normalize_country
    )

    return df


def main():

    print("=" * 80)
    print("BLOCKING RECALL TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load 5,000 Source 1 entities
    # ---------------------------------------------------------

    source1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        nrows=5000,
        dtype=str
    )

    source1_ids = set(source1["entity_id"])

    print(f"\nSource 1 sample: {len(source1):,}")

    # ---------------------------------------------------------
    # 2. Load ground truth
    # ---------------------------------------------------------

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str
    )

    gt = gt[
        gt["source1_entity_id"].isin(source1_ids)
    ].copy()

    true_pairs = set()

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        if not str(matched_ids).strip():
            continue

        for s2_id in str(matched_ids).split(","):

            s2_id = s2_id.strip()

            if s2_id.startswith("S2-"):
                true_pairs.add(
                    (s1_id, s2_id)
                )

    print(f"True S1 -> S2 pairs: {len(true_pairs):,}")

    # ---------------------------------------------------------
    # 3. Collect required Source 2 records
    # ---------------------------------------------------------

    required_s2_ids = {
        pair[1]
        for pair in true_pairs
    }

    print(
        f"Unique true S2 entities needed: "
        f"{len(required_s2_ids):,}"
    )

    required_s2 = []

    for chunk in pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        chunksize=100000
    ):

        matched = chunk[
            chunk["entity_id"].isin(required_s2_ids)
        ]

        if not matched.empty:
            required_s2.append(matched)

        current_count = sum(
            len(x) for x in required_s2
        )

        if current_count >= len(required_s2_ids):
            break

    source2 = pd.concat(
        required_s2,
        ignore_index=True
    )

    print(
        f"S2 records loaded for recall test: "
        f"{len(source2):,}"
    )

    # ---------------------------------------------------------
    # 4. Normalize data
    # ---------------------------------------------------------

    source1 = prepare_source(source1)
    source2 = prepare_source(source2)

    # ---------------------------------------------------------
    # 5. Generate candidates
    # ---------------------------------------------------------

    candidates = generate_candidates(
        source1,
        source2
    )

    candidate_pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["matched_entity_id"]
        )
    )

    print(
        f"\nGenerated candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    # ---------------------------------------------------------
    # 6. Calculate blocking recall
    # ---------------------------------------------------------

    recovered = true_pairs.intersection(
        candidate_pairs
    )

    missed = true_pairs - candidate_pairs

    if true_pairs:

        recall = (
            len(recovered)
            / len(true_pairs)
            * 100
        )

    else:
        recall = 0

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"True pairs:       {len(true_pairs):,}")
    print(f"Recovered pairs:  {len(recovered):,}")
    print(f"Missed pairs:     {len(missed):,}")
    print(f"Blocking recall:  {recall:.2f}%")

    # ---------------------------------------------------------
    # 7. Show missed examples
    # ---------------------------------------------------------

    if missed:

        print("\nSample missed true pairs:")

        for s1_id, s2_id in list(missed)[:20]:

            print(
                f"  {s1_id} -> {s2_id}"
            )

    else:

        print(
            "\nExcellent: no true pairs were missed."
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()