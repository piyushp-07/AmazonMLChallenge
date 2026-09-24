import pandas as pd
import re
from collections import defaultdict

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN_DIR = f"{DATASET_ROOT}\\dataset\\train"

S1_SAMPLE_SIZE = 5000
S2_CHUNK_SIZE = 100_000


# ============================================================
# LOAD DATA
# ============================================================

def load_source1():

    path = f"{TRAIN_DIR}\\train_source1.tsv"

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=S1_SAMPLE_SIZE,
        dtype=str,
    ).fillna("")

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["country_norm"] = df["country"].map(
        normalize_country
    )

    return df


def load_ground_truth(source1_ids):

    path = f"{TRAIN_DIR}\\train_ground_truth.tsv"

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    ).fillna("")

    gt = gt[
        gt["source1_entity_id"].isin(source1_ids)
    ]

    true_pairs = set()

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]

        for matched_id in row[
            "matched_entity_ids"
        ].split(","):

            matched_id = matched_id.strip()

            if matched_id.startswith("S2-"):

                true_pairs.add(
                    (
                        s1_id,
                        matched_id,
                    )
                )

    return true_pairs


def load_required_source2(true_pairs):

    required_ids = {
        s2_id
        for _, s2_id in true_pairs
    }

    path = f"{TRAIN_DIR}\\train_source2.tsv"

    rows = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=S2_CHUNK_SIZE,
    ):

        chunk = chunk.fillna("")

        matched = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if not matched.empty:
            rows.append(matched)

    df = pd.concat(
        rows,
        ignore_index=True,
    )

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["country_norm"] = df["country"].map(
        normalize_country
    )

    return df


# ============================================================
# NAME TOKENIZATION
# ============================================================

NAME_STOPWORDS = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "private",
    "pvt",
    "plc",
    "corp",
    "corporation",
    "company",
    "co",
    "llp",
    "pc",
    "services",
    "service",
    "center",
    "centre",
    "group",
}


def get_name_tokens(name):

    if not name:
        return []

    tokens = re.findall(
        r"[a-z0-9]+",
        name.lower(),
    )

    useful = []

    for token in tokens:

        if token in NAME_STOPWORDS:
            continue

        if len(token) < 3:
            continue

        useful.append(token)

    return list(
        dict.fromkeys(useful)
    )


# ============================================================
# TOKEN BLOCKING
# ============================================================

def generate_token_candidates(
    source1,
    source2,
):

    index = defaultdict(set)

    # --------------------------------------------------------
    # Build S2 index
    # --------------------------------------------------------

    for _, row in source2.iterrows():

        s2_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]

        if not country or not name:
            continue

        tokens = get_name_tokens(name)

        for token in tokens:

            index[
                (
                    country,
                    token,
                )
            ].add(s2_id)

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidates = set()

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]

        if not country or not name:
            continue

        tokens = get_name_tokens(name)

        for token in tokens:

            for s2_id in index.get(
                (
                    country,
                    token,
                ),
                set(),
            ):

                candidates.add(
                    (
                        s1_id,
                        s2_id,
                    )
                )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NAME TOKEN BLOCKING TEST")
    print("=" * 70)

    source1 = load_source1()

    source1_ids = set(
        source1["entity_id"]
    )

    true_pairs = load_ground_truth(
        source1_ids
    )

    source2 = load_required_source2(
        true_pairs
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

    print("\nGenerating token candidates...")

    candidates = generate_token_candidates(
        source1,
        source2,
    )

    recovered = (
        true_pairs
        & candidates
    )

    missed = (
        true_pairs
        - candidates
    )

    recall = (
        len(recovered)
        / len(true_pairs)
        * 100
    )

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"Token candidates:     "
        f"{len(candidates):,}"
    )

    print(
        f"Average candidates/S1: "
        f"{len(candidates) / len(source1):.2f}"
    )

    print(
        f"Recovered pairs:      "
        f"{len(recovered):,}"
    )

    print(
        f"Missed pairs:         "
        f"{len(missed):,}"
    )

    print(
        f"Token blocking recall: "
        f"{recall:.2f}%"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()