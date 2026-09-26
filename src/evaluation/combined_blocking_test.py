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
# LOAD SOURCE 1
# ============================================================

def load_sample_source1():

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
        normalize_country
    )

    return df


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(source1_ids):

    path = f"{TRAIN_DIR}\\train_ground_truth.tsv"

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    ).fillna("")

    gt = gt[
        gt["source1_entity_id"].isin(source1_ids)
    ].copy()

    true_pairs = set()

    for _, row in gt.iterrows():

        source1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for matched_id in matched_ids.split(","):

            matched_id = matched_id.strip()

            if matched_id:
                true_pairs.add(
                    (source1_id, matched_id)
                )

    return true_pairs


# ============================================================
# LOAD REQUIRED SOURCE 2 RECORDS
# ============================================================

def load_required_source2(true_pairs):

    required_ids = {
        matched_id
        for _, matched_id in true_pairs
    }

    path = f"{TRAIN_DIR}\\train_source2.tsv"

    required_rows = []

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
            required_rows.append(matched)

    if not required_rows:
        return pd.DataFrame()

    df = pd.concat(
        required_rows,
        ignore_index=True,
    )

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


# ============================================================
# NAME BLOCKING
# ============================================================

def name_prefix(value, length=4):

    if not value:
        return ""

    value = value.strip()

    if len(value) < length:
        return value

    return value[:length]


def create_name_blocking_indexes(source2):

    indexes = {
        "exact_name_country": defaultdict(set),
        "prefix4_country": defaultdict(set),
        "prefix6_country": defaultdict(set),
    }

    for _, row in source2.iterrows():

        entity_id = row["entity_id"]
        name = row["name_norm"]
        country = row["country_norm"]

        if not name or not country:
            continue

        # Rule 1:
        # Exact normalized name + country

        indexes["exact_name_country"][
            (country, name)
        ].add(entity_id)

        # Rule 2:
        # Country + first 4 characters

        prefix4 = name_prefix(
            name,
            4
        )

        if prefix4:

            indexes["prefix4_country"][
                (country, prefix4)
            ].add(entity_id)

        # Rule 3:
        # Country + first 6 characters

        prefix6 = name_prefix(
            name,
            6
        )

        if prefix6:

            indexes["prefix6_country"][
                (country, prefix6)
            ].add(entity_id)

    return indexes


def generate_name_candidates(source1, source2):

    indexes = create_name_blocking_indexes(
        source2
    )

    candidates = set()

    for _, row in source1.iterrows():

        source1_id = row["entity_id"]
        name = row["name_norm"]
        country = row["country_norm"]

        if not name or not country:
            continue

        # ----------------------------------------------------
        # Exact name + country
        # ----------------------------------------------------

        key = (
            country,
            name,
        )

        for source2_id in indexes[
            "exact_name_country"
        ].get(key, set()):

            candidates.add(
                (
                    source1_id,
                    source2_id,
                )
            )

        # ----------------------------------------------------
        # 4-character prefix + country
        # ----------------------------------------------------

        prefix4 = name_prefix(
            name,
            4
        )

        if prefix4:

            key = (
                country,
                prefix4,
            )

            for source2_id in indexes[
                "prefix4_country"
            ].get(key, set()):

                candidates.add(
                    (
                        source1_id,
                        source2_id,
                    )
                )

        # ----------------------------------------------------
        # 6-character prefix + country
        # ----------------------------------------------------

        prefix6 = name_prefix(
            name,
            6
        )

        if prefix6:

            key = (
                country,
                prefix6,
            )

            for source2_id in indexes[
                "prefix6_country"
            ].get(key, set()):

                candidates.add(
                    (
                        source1_id,
                        source2_id,
                    )
                )

    return candidates


# ============================================================
# ADDRESS TOKENIZATION
# ============================================================

def get_address_tokens(address):

    if not address:
        return []

    tokens = re.findall(
        r"[a-z0-9]+",
        address.lower(),
    )

    stopwords = {
        "road",
        "rd",
        "street",
        "st",
        "avenue",
        "ave",
        "lane",
        "ln",
        "drive",
        "dr",
        "highway",
        "hwy",
        "the",
        "of",
        "and",
        "near",
        "opp",
        "opposite",
        "block",
        "area",
        "district",
        "city",
        "county",
        "state",
        "india",
        "usa",
        "us",
    }

    useful = []

    for token in tokens:

        if token in stopwords:
            continue

        if len(token) < 3:
            continue

        useful.append(token)

    return list(
        dict.fromkeys(useful)
    )


def get_address_numbers(address):

    if not address:
        return []

    return re.findall(
        r"\d+",
        address,
    )


# ============================================================
# ADDRESS BLOCKING
# ============================================================

def create_address_blocking_indexes(source2):

    indexes = {
        "exact_address": defaultdict(set),
        "number_token": defaultdict(set),
        "token_pair": defaultdict(set),
    }

    for _, row in source2.iterrows():

        entity_id = row["entity_id"]
        country = row["country_norm"]
        address = row["address_norm"]

        if not country or not address:
            continue

        useful_tokens = get_address_tokens(
            address
        )

        numbers = get_address_numbers(
            address
        )

        # ----------------------------------------------------
        # Rule 1: Exact address + country
        # ----------------------------------------------------

        indexes["exact_address"][
            (
                country,
                address,
            )
        ].add(entity_id)

        # ----------------------------------------------------
        # Rule 2: Number + token + country
        # ----------------------------------------------------

        for number in numbers:

            for token in useful_tokens:

                indexes["number_token"][
                    (
                        country,
                        number,
                        token,
                    )
                ].add(entity_id)

        # ----------------------------------------------------
        # Rule 3: Two address tokens + country
        # ----------------------------------------------------

        if len(useful_tokens) >= 2:

            for i in range(
                len(useful_tokens)
            ):

                for j in range(
                    i + 1,
                    len(useful_tokens),
                ):

                    token1 = useful_tokens[i]
                    token2 = useful_tokens[j]

                    pair = tuple(
                        sorted(
                            (
                                token1,
                                token2,
                            )
                        )
                    )

                    indexes["token_pair"][
                        (
                            country,
                            pair,
                        )
                    ].add(entity_id)

    return indexes


def generate_address_candidates(
    source1,
    source2,
):

    indexes = create_address_blocking_indexes(
        source2
    )

    candidates = set()

    for _, row in source1.iterrows():

        source1_id = row["entity_id"]
        country = row["country_norm"]
        address = row["address_norm"]

        if not country or not address:
            continue

        useful_tokens = get_address_tokens(
            address
        )

        numbers = get_address_numbers(
            address
        )

        # ----------------------------------------------------
        # Rule 1
        # ----------------------------------------------------

        exact_key = (
            country,
            address,
        )

        for source2_id in indexes[
            "exact_address"
        ].get(
            exact_key,
            set(),
        ):

            candidates.add(
                (
                    source1_id,
                    source2_id,
                )
            )

        # ----------------------------------------------------
        # Rule 2
        # ----------------------------------------------------

        for number in numbers:

            for token in useful_tokens:

                key = (
                    country,
                    number,
                    token,
                )

                for source2_id in indexes[
                    "number_token"
                ].get(
                    key,
                    set(),
                ):

                    candidates.add(
                        (
                            source1_id,
                            source2_id,
                        )
                    )

        # ----------------------------------------------------
        # Rule 3
        # ----------------------------------------------------

        if len(useful_tokens) >= 2:

            for i in range(
                len(useful_tokens)
            ):

                for j in range(
                    i + 1,
                    len(useful_tokens),
                ):

                    token1 = useful_tokens[i]
                    token2 = useful_tokens[j]

                    pair = tuple(
                        sorted(
                            (
                                token1,
                                token2,
                            )
                        )
                    )

                    key = (
                        country,
                        pair,
                    )

                    for source2_id in indexes[
                        "token_pair"
                    ].get(
                        key,
                        set(),
                    ):

                        candidates.add(
                            (
                                source1_id,
                                source2_id,
                            )
                        )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COMBINED NAME + ADDRESS BLOCKING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Source 1
    # --------------------------------------------------------

    source1 = load_sample_source1()

    source1_ids = set(
        source1["entity_id"]
    )

    print(
        f"\nSource 1 sample: "
        f"{len(source1):,}"
    )

    # --------------------------------------------------------
    # 2. Ground truth
    # --------------------------------------------------------

    true_pairs = load_ground_truth(
        source1_ids
    )

    print(
        f"True S1 -> S2 pairs: "
        f"{len(true_pairs):,}"
    )

    # --------------------------------------------------------
    # 3. Required Source 2 records
    # --------------------------------------------------------

    source2 = load_required_source2(
        true_pairs
    )

    required_s2_ids = {
        matched_id
        for _, matched_id in true_pairs
    }

    loaded_s2_ids = set(
        source2["entity_id"]
    )

    evaluable_true_pairs = {
        pair
        for pair in true_pairs
        if pair[1] in loaded_s2_ids
    }

    print(
        f"Unique true S2 entities needed: "
        f"{len(required_s2_ids):,}"
    )

    print(
        f"S2 records loaded: "
        f"{len(source2):,}"
    )

    print(
        f"Evaluable true pairs: "
        f"{len(evaluable_true_pairs):,}"
    )

    # --------------------------------------------------------
    # 4. Name candidates
    # --------------------------------------------------------

    print("\nGenerating name candidates...")

    name_candidates = generate_name_candidates(
        source1,
        source2,
    )

    # --------------------------------------------------------
    # 5. Address candidates
    # --------------------------------------------------------

    print(
        "Generating address candidates..."
    )

    address_candidates = generate_address_candidates(
        source1,
        source2,
    )

    # --------------------------------------------------------
    # 6. Combined candidates
    # --------------------------------------------------------

    combined_candidates = (
        name_candidates
        | address_candidates
    )

    # --------------------------------------------------------
    # 7. Recall
    # --------------------------------------------------------

    name_recovered = (
        evaluable_true_pairs
        .intersection(name_candidates)
    )

    address_recovered = (
        evaluable_true_pairs
        .intersection(address_candidates)
    )

    combined_recovered = (
        evaluable_true_pairs
        .intersection(combined_candidates)
    )

    combined_missed = (
        evaluable_true_pairs
        - combined_candidates
    )

    name_recall = (
        len(name_recovered)
        / len(evaluable_true_pairs)
        * 100
        if evaluable_true_pairs
        else 0
    )

    address_recall = (
        len(address_recovered)
        / len(evaluable_true_pairs)
        * 100
        if evaluable_true_pairs
        else 0
    )

    combined_recall = (
        len(combined_recovered)
        / len(evaluable_true_pairs)
        * 100
        if evaluable_true_pairs
        else 0
    )

    # --------------------------------------------------------
    # 8. Candidate statistics
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"Name candidates:       "
        f"{len(name_candidates):,}"
    )

    print(
        f"Address candidates:    "
        f"{len(address_candidates):,}"
    )

    print(
        f"Combined candidates:   "
        f"{len(combined_candidates):,}"
    )

    print(
        f"Average candidates/S1: "
        f"{len(combined_candidates) / len(source1):.2f}"
    )

    print()

    print(
        f"Name recovered:        "
        f"{len(name_recovered):,}"
    )

    print(
        f"Name recall:           "
        f"{name_recall:.2f}%"
    )

    print()

    print(
        f"Address recovered:     "
        f"{len(address_recovered):,}"
    )

    print(
        f"Address recall:        "
        f"{address_recall:.2f}%"
    )

    print()

    print(
        f"Combined recovered:    "
        f"{len(combined_recovered):,}"
    )

    print(
        f"Combined missed:       "
        f"{len(combined_missed):,}"
    )

    print(
        f"Combined recall:       "
        f"{combined_recall:.2f}%"
    )

    # --------------------------------------------------------
    # 9. Contribution analysis
    # --------------------------------------------------------

    address_only_recovery = (
        address_recovered
        - name_recovered
    )

    name_only_recovery = (
        name_recovered
        - address_recovered
    )

    both_recovery = (
        name_recovered
        & address_recovered
    )

    print("\n" + "-" * 70)
    print("BLOCKING CONTRIBUTION")
    print("-" * 70)

    print(
        f"Recovered by NAME only:     "
        f"{len(name_only_recovery):,}"
    )

    print(
        f"Recovered by ADDRESS only:  "
        f"{len(address_only_recovery):,}"
    )

    print(
        f"Recovered by BOTH:          "
        f"{len(both_recovery):,}"
    )

    # --------------------------------------------------------
    # 10. Sample combined misses
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("SAMPLE STILL-MISSED PAIRS")
    print("-" * 70)

    for source1_id, source2_id in list(
        combined_missed
    )[:10]:

        s1_matches = source1[
            source1["entity_id"]
            == source1_id
        ]

        s2_matches = source2[
            source2["entity_id"]
            == source2_id
        ]

        if s1_matches.empty:
            continue

        if s2_matches.empty:
            continue

        s1_row = s1_matches.iloc[0]
        s2_row = s2_matches.iloc[0]

        print(
            f"\nS1: {source1_id}"
        )

        print(
            f"    Name:    "
            f"{s1_row['business_name']}"
        )

        print(
            f"    Address: "
            f"{s1_row['business_address']}"
        )

        print(
            f"S2: {source2_id}"
        )

        print(
            f"    Name:    "
            f"{s2_row['business_name']}"
        )

        print(
            f"    Address: "
            f"{s2_row['business_address']}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()