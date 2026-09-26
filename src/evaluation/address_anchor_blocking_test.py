import os
import re
import pandas as pd

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address
)


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN_DIR = os.path.join(DATASET_ROOT, "dataset", "train")


def load_ground_truth(s1_ids):

    gt_path = os.path.join(
        TRAIN_DIR,
        "train_ground_truth.tsv"
    )

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str
    )

    gt = gt[
        gt["source1_entity_id"].isin(s1_ids)
    ]

    true_pairs = set()

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        matched = str(row["matched_entity_ids"])

        if matched == "nan" or not matched.strip():
            continue

        for matched_id in matched.split(","):

            matched_id = matched_id.strip()

            if matched_id.startswith("S2-"):
                true_pairs.add(
                    (s1_id, matched_id)
                )

    return true_pairs


def load_required_source2(required_ids):

    path = os.path.join(
        TRAIN_DIR,
        "train_source2.tsv"
    )

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

    return pd.concat(
        chunks,
        ignore_index=True
    )


def normalize_dataframe(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].apply(
        normalize_business_name
    )

    df["address_norm"] = df["business_address"].apply(
        normalize_address
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def extract_address_anchors(address):

    if not address:
        return set()

    address = address.lower()

    anchors = set()

    # Normalize common separators
    address = re.sub(
        r"[-/,#().]+",
        " ",
        address
    )

    tokens = address.split()

    for token in tokens:

        token = token.strip()

        if not token:
            continue

        # Pure numbers: 2-6 digits
        if re.fullmatch(r"\d{2,6}", token):
            anchors.add(token)

        # Alphanumeric house/address tokens:
        # 46d, 57b, a3, 121a, etc.
        elif re.fullmatch(
            r"[a-z]{1,3}\d{1,5}|\d{1,5}[a-z]{1,3}",
            token
        ):
            anchors.add(token)

    return anchors


def build_anchor_candidates(source1, source2):

    index = {}

    for _, row in source2.iterrows():

        country = row["country_norm"]
        address = row["address_norm"]

        if not country or not address:
            continue

        anchors = extract_address_anchors(
            address
        )

        for anchor in anchors:

            key = (
                country,
                anchor
            )

            index.setdefault(
                key,
                set()
            ).add(
                row["entity_id"]
            )

    candidates = set()

    for _, row in source1.iterrows():

        country = row["country_norm"]
        address = row["address_norm"]

        if not country or not address:
            continue

        anchors = extract_address_anchors(
            address
        )

        for anchor in anchors:

            key = (
                country,
                anchor
            )

            for s2_id in index.get(
                key,
                set()
            ):

                candidates.add(
                    (
                        row["entity_id"],
                        s2_id
                    )
                )

    return candidates


def evaluate(
    candidates,
    true_pairs
):

    recovered = (
        candidates &
        true_pairs
    )

    missed = (
        true_pairs -
        candidates
    )

    recall = (
        len(recovered) /
        len(true_pairs)
        if true_pairs
        else 0
    )

    print(
        f"Anchor candidates:     "
        f"{len(candidates):,}"
    )

    print(
        f"Average candidates/S1: "
        f"{len(candidates) / 5000:.2f}"
    )

    print(
        f"Recovered pairs:       "
        f"{len(recovered):,}"
    )

    print(
        f"Missed pairs:           "
        f"{len(missed):,}"
    )

    print(
        f"Anchor blocking recall: "
        f"{recall:.2%}"
    )

    return recovered, missed


def main():

    print("=" * 70)
    print("ADDRESS ANCHOR BLOCKING TEST")
    print("=" * 70)

    source1_path = os.path.join(
        TRAIN_DIR,
        "train_source1.tsv"
    )

    source1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        nrows=5000
    )

    source1 = normalize_dataframe(
        source1
    )

    true_pairs = load_ground_truth(
        set(source1["entity_id"])
    )

    required_s2_ids = {
        s2_id
        for _, s2_id in true_pairs
    }

    source2 = load_required_source2(
        required_s2_ids
    )

    source2 = normalize_dataframe(
        source2
    )

    print(
        f"\nSource 1 rows: "
        f"{len(source1):,}"
    )

    print(
        f"Source 2 rows: "
        f"{len(source2):,}"
    )

    print(
        f"True S1 -> S2 pairs: "
        f"{len(true_pairs):,}"
    )

    print("\nGenerating address-anchor candidates...")

    candidates = build_anchor_candidates(
        source1,
        source2
    )

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)

    recovered, missed = evaluate(
        candidates,
        true_pairs
    )

    print("\n" + "-" * 70)
    print("MISSED TRUE PAIRS")
    print("-" * 70)

    for pair in sorted(missed):

        print(
            f"{pair[0]} -> {pair[1]}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
