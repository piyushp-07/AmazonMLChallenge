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

    print("=" * 100)
    print("MISSED TRUE PAIR ANALYSIS")
    print("=" * 100)

    # ---------------------------------------------------------
    # 1. Load 5,000 Source 1 records
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
    # 2. Load ground truth for these S1 records
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
    # 3. Load required Source 2 records
    # ---------------------------------------------------------

    required_s2_ids = {
        pair[1]
        for pair in true_pairs
    }

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
        f"S2 records loaded: {len(source2):,}"
    )

    # ---------------------------------------------------------
    # 4. Normalize
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

    # ---------------------------------------------------------
    # 6. Find missed pairs
    # ---------------------------------------------------------

    missed = true_pairs - candidate_pairs

    print(
        f"Missed true pairs: {len(missed):,}"
    )

    print("\n" + "=" * 100)
    print("FIRST 50 MISSED TRUE PAIRS")
    print("=" * 100)

    # Create lookup tables
    s1_lookup = source1.set_index("entity_id")
    s2_lookup = source2.set_index("entity_id")

    for index, (s1_id, s2_id) in enumerate(
        list(missed)[:50],
        start=1
    ):

        s1 = s1_lookup.loc[s1_id]
        s2 = s2_lookup.loc[s2_id]

        print("\n" + "-" * 100)
        print(f"MISSED PAIR #{index}")
        print("-" * 100)

        print(f"S1 ID:       {s1_id}")
        print(f"S2 ID:       {s2_id}")
        print(f"Country:     {s1['country']}")

        print("\nSOURCE 1")
        print(f"Name:        {s1['business_name']}")
        print(f"Address:     {s1['business_address']}")

        print("\nSOURCE 2")
        print(f"Name:        {s2['business_name']}")
        print(f"Address:     {s2['business_address']}")

        print("\nNORMALIZED")
        print(f"S1 name:     {s1['name_norm']}")
        print(f"S2 name:     {s2['name_norm']}")
        print(f"S1 address:  {s1['address_norm']}")
        print(f"S2 address:  {s2['address_norm']}")

    print("\n" + "=" * 100)
    print("END OF ANALYSIS")
    print("=" * 100)


if __name__ == "__main__":
    main()