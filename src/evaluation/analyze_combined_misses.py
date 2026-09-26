import pandas as pd
import re

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN_DIR = f"{DATASET_ROOT}\\dataset\\train"

S1_SAMPLE_SIZE = 5000
S2_CHUNK_SIZE = 100_000


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

    df["address_norm"] = df["business_address"].map(
        normalize_address
    )

    df["country_norm"] = df["country"].map(
        lambda x: str(x).strip().lower()
        if x else ""
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

            # IMPORTANT:
            # Only Source 2 IDs.
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

    df["address_norm"] = df["business_address"].map(
        normalize_address
    )

    df["country_norm"] = df["country"].map(
        lambda x: str(x).strip().lower()
        if x else ""
    )

    return df


# ============================================================
# SAME BLOCKING RULES
# ============================================================

def name_prefix(value, length):

    if not value:
        return ""

    return value[:length]


def get_address_tokens(address):

    if not address:
        return []

    tokens = re.findall(
        r"[a-z0-9]+",
        address.lower(),
    )

    stopwords = {
        "road", "rd",
        "street", "st",
        "avenue", "ave",
        "lane", "ln",
        "drive", "dr",
        "highway", "hwy",
        "the", "of", "and",
        "near", "opp", "opposite",
        "block", "area",
        "district", "city",
        "county", "state",
        "india", "usa", "us",
    }

    return list(
        dict.fromkeys(
            token
            for token in tokens
            if token not in stopwords
            and len(token) >= 3
        )
    )


def get_address_numbers(address):

    if not address:
        return []

    return re.findall(
        r"\d+",
        address,
    )


def generate_name_candidates(source1, source2):

    exact_index = {}
    prefix4_index = {}
    prefix6_index = {}

    for _, row in source2.iterrows():

        s2_id = row["entity_id"]
        name = row["name_norm"]
        country = row["country_norm"]

        if not name or not country:
            continue

        exact_index.setdefault(
            (country, name),
            set(),
        ).add(s2_id)

        prefix4_index.setdefault(
            (
                country,
                name_prefix(name, 4),
            ),
            set(),
        ).add(s2_id)

        prefix6_index.setdefault(
            (
                country,
                name_prefix(name, 6),
            ),
            set(),
        ).add(s2_id)

    candidates = set()

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]
        name = row["name_norm"]
        country = row["country_norm"]

        if not name or not country:
            continue

        for s2_id in exact_index.get(
            (country, name),
            set(),
        ):
            candidates.add(
                (s1_id, s2_id)
            )

        for s2_id in prefix4_index.get(
            (
                country,
                name_prefix(name, 4),
            ),
            set(),
        ):
            candidates.add(
                (s1_id, s2_id)
            )

        for s2_id in prefix6_index.get(
            (
                country,
                name_prefix(name, 6),
            ),
            set(),
        ):
            candidates.add(
                (s1_id, s2_id)
            )

    return candidates


def generate_address_candidates(source1, source2):

    exact_index = {}
    number_token_index = {}
    token_pair_index = {}

    for _, row in source2.iterrows():

        s2_id = row["entity_id"]
        country = row["country_norm"]
        address = row["address_norm"]

        if not country or not address:
            continue

        tokens = get_address_tokens(address)
        numbers = get_address_numbers(address)

        exact_index.setdefault(
            (country, address),
            set(),
        ).add(s2_id)

        for number in numbers:

            for token in tokens:

                number_token_index.setdefault(
                    (
                        country,
                        number,
                        token,
                    ),
                    set(),
                ).add(s2_id)

        for i in range(len(tokens)):

            for j in range(i + 1, len(tokens)):

                pair = tuple(
                    sorted(
                        (
                            tokens[i],
                            tokens[j],
                        )
                    )
                )

                token_pair_index.setdefault(
                    (
                        country,
                        pair,
                    ),
                    set(),
                ).add(s2_id)

    candidates = set()

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]
        country = row["country_norm"]
        address = row["address_norm"]

        if not country or not address:
            continue

        tokens = get_address_tokens(address)
        numbers = get_address_numbers(address)

        for s2_id in exact_index.get(
            (country, address),
            set(),
        ):
            candidates.add(
                (s1_id, s2_id)
            )

        for number in numbers:

            for token in tokens:

                for s2_id in number_token_index.get(
                    (
                        country,
                        number,
                        token,
                    ),
                    set(),
                ):

                    candidates.add(
                        (s1_id, s2_id)
                    )

        for i in range(len(tokens)):

            for j in range(i + 1, len(tokens)):

                pair = tuple(
                    sorted(
                        (
                            tokens[i],
                            tokens[j],
                        )
                    )
                )

                for s2_id in token_pair_index.get(
                    (
                        country,
                        pair,
                    ),
                    set(),
                ):

                    candidates.add(
                        (s1_id, s2_id)
                    )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading data...")

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
        f"S1 rows: {len(source1):,}"
    )

    print(
        f"S2 rows loaded: {len(source2):,}"
    )

    print(
        f"True S1-S2 pairs: {len(true_pairs):,}"
    )

    print("\nGenerating blocking candidates...")

    name_candidates = generate_name_candidates(
        source1,
        source2,
    )

    address_candidates = generate_address_candidates(
        source1,
        source2,
    )

    combined_candidates = (
        name_candidates
        | address_candidates
    )

    missed = (
        true_pairs
        - combined_candidates
    )

    print(
        f"\nCombined missed pairs: "
        f"{len(missed)}"
    )

    print("\n" + "=" * 80)
    print("ALL MISSED PAIRS")
    print("=" * 80)

    for number, (s1_id, s2_id) in enumerate(
        sorted(missed),
        start=1,
    ):

        s1 = source1[
            source1["entity_id"]
            == s1_id
        ]

        s2 = source2[
            source2["entity_id"]
            == s2_id
        ]

        if s1.empty or s2.empty:
            continue

        s1 = s1.iloc[0]
        s2 = s2.iloc[0]

        print("\n" + "-" * 80)

        print(f"MISSED #{number}")

        print(f"\nS1 ID: {s1_id}")
        print(
            f"S1 Name:    "
            f"{s1['business_name']}"
        )
        print(
            f"S1 Address: "
            f"{s1['business_address']}"
        )
        print(
            f"S1 Country: "
            f"{s1['country']}"
        )

        print(f"\nS2 ID: {s2_id}")
        print(
            f"S2 Name:    "
            f"{s2['business_name']}"
        )
        print(
            f"S2 Address: "
            f"{s2['business_address']}"
        )
        print(
            f"S2 Country: "
            f"{s2['country']}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()