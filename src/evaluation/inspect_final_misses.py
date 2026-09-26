import os
import pandas as pd

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address
)


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN_DIR = os.path.join(DATASET_ROOT, "dataset", "train")


MISSED_PAIRS = [
    ("S1-163077720", "S2-906538643"),
    ("S1-239986792", "S2-615936168"),
    ("S1-289445281", "S2-708881630"),
    ("S1-394340130", "S2-802580138"),
    ("S1-669413769", "S2-805282132"),
    ("S1-903445032", "S2-689432168"),
    ("S1-967735640", "S2-155233311"),
    ("S1-987010400", "S2-140661931"),
]


def load_records(path, required_ids):
    chunks = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100_000
    ):
        found = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if not found.empty:
            chunks.append(found)

        if sum(len(x) for x in chunks) >= len(required_ids):
            break

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def main():

    print("=" * 80)
    print("INSPECTING FINAL 8 MISSED PAIRS")
    print("=" * 80)

    s1_ids = {
        pair[0]
        for pair in MISSED_PAIRS
    }

    s2_ids = {
        pair[1]
        for pair in MISSED_PAIRS
    }

    s1_path = os.path.join(
        TRAIN_DIR,
        "train_source1.tsv"
    )

    s2_path = os.path.join(
        TRAIN_DIR,
        "train_source2.tsv"
    )

    source1 = load_records(
        s1_path,
        s1_ids
    )

    source2 = load_records(
        s2_path,
        s2_ids
    )

    source1 = source1.set_index("entity_id")
    source2 = source2.set_index("entity_id")

    for i, (s1_id, s2_id) in enumerate(
        MISSED_PAIRS,
        start=1
    ):

        s1 = source1.loc[s1_id]
        s2 = source2.loc[s2_id]

        print("\n" + "=" * 80)
        print(f"MISSED PAIR {i}/8")
        print("=" * 80)

        print(f"S1 ID: {s1_id}")
        print(f"S2 ID: {s2_id}")

        print("\nSOURCE 1")
        print("-" * 80)
        print(f"Name:    {s1['business_name']}")
        print(f"Address: {s1['business_address']}")
        print(f"Country: {s1['country']}")

        print("\nNORMALIZED S1")
        print("-" * 80)
        print(
            "Name:    "
            f"{normalize_business_name(s1['business_name'])}"
        )
        print(
            "Address: "
            f"{normalize_address(s1['business_address'])}"
        )

        print("\nSOURCE 2")
        print("-" * 80)
        print(f"Name:    {s2['business_name']}")
        print(f"Address: {s2['business_address']}")
        print(f"Country: {s2['country']}")

        print("\nNORMALIZED S2")
        print("-" * 80)
        print(
            "Name:    "
            f"{normalize_business_name(s2['business_name'])}"
        )
        print(
            "Address: "
            f"{normalize_address(s2['business_address'])}"
        )

    print("\n" + "=" * 80)
    print("END")
    print("=" * 80)


if __name__ == "__main__":
    main()