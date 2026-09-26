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


def load_sample_source1():
    path = f"{TRAIN_DIR}\\train_source1.tsv"

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=S1_SAMPLE_SIZE,
        dtype=str,
    ).fillna("")

    df["name_norm"] = df["business_name"].map(normalize_business_name)
    df["address_norm"] = df["business_address"].map(normalize_address)
    df["country_norm"] = df["country"].map(normalize_country)

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


def load_required_source2(true_pairs):
    """
    Load every Source 2 record required by the ground truth.

    We scan the complete Source 2 file because the required
    records may appear anywhere in the file.
    """

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


def get_address_tokens(address):
    """
    Extract useful address tokens.

    Common address words are removed because they can
    create very large candidate blocks.
    """

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

    return list(dict.fromkeys(useful))


def get_address_numbers(address):
    """
    Extract numeric components from an address.

    Examples:
        102 Ruby Avenue -> ['102']
        A-68 New Delhi -> ['68']
    """

    if not address:
        return []

    return re.findall(
        r"\d+",
        address,
    )


def build_address_candidates(source1, source2):
    """
    Generate address-based candidates using three rules.

    Rule 1:
        Exact normalized address + country

    Rule 2:
        Country + address number + useful address token

    Rule 3:
        Country + two useful address tokens
    """

    indexes = {
        "exact_address": defaultdict(set),
        "number_token": defaultdict(set),
        "token_pair": defaultdict(set),
    }

    # =========================================================
    # BUILD SOURCE 2 INDEXES
    # =========================================================

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

        # -----------------------------------------------------
        # Rule 1: exact address + country
        # -----------------------------------------------------

        indexes["exact_address"][
            (country, address)
        ].add(entity_id)

        # -----------------------------------------------------
        # Rule 2: number + useful token + country
        # -----------------------------------------------------

        for number in numbers:

            for token in useful_tokens:

                indexes["number_token"][
                    (country, number, token)
                ].add(entity_id)

        # -----------------------------------------------------
        # Rule 3: two useful tokens + country
        # -----------------------------------------------------

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
                            (token1, token2)
                        )
                    )

                    indexes["token_pair"][
                        (country, pair)
                    ].add(entity_id)

    candidates = set()

    # =========================================================
    # GENERATE SOURCE 1 -> SOURCE 2 CANDIDATES
    # =========================================================

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

        # -----------------------------------------------------
        # Rule 1
        # -----------------------------------------------------

        exact_key = (
            country,
            address,
        )

        for source2_id in indexes[
            "exact_address"
        ].get(exact_key, set()):

            candidates.add(
                (
                    source1_id,
                    source2_id,
                )
            )

        # -----------------------------------------------------
        # Rule 2
        # -----------------------------------------------------

        for number in numbers:

            for token in useful_tokens:

                key = (
                    country,
                    number,
                    token,
                )

                for source2_id in indexes[
                    "number_token"
                ].get(key, set()):

                    candidates.add(
                        (
                            source1_id,
                            source2_id,
                        )
                    )

        # -----------------------------------------------------
        # Rule 3
        # -----------------------------------------------------

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
                            (token1, token2)
                        )
                    )

                    key = (
                        country,
                        pair,
                    )

                    for source2_id in indexes[
                        "token_pair"
                    ].get(key, set()):

                        candidates.add(
                            (
                                source1_id,
                                source2_id,
                            )
                        )

    return candidates


def main():

    print("=" * 70)
    print("ADDRESS BLOCKING RECALL TEST")
    print("=" * 70)

    # =========================================================
    # 1. LOAD SOURCE 1 SAMPLE
    # =========================================================

    source1 = load_sample_source1()

    source1_ids = set(
        source1["entity_id"]
    )

    print(
        f"\nSource 1 sample: "
        f"{len(source1):,}"
    )

    # =========================================================
    # 2. LOAD GROUND TRUTH
    # =========================================================

    true_pairs = load_ground_truth(
        source1_ids
    )

    print(
        f"True S1 -> S2 pairs: "
        f"{len(true_pairs):,}"
    )

    required_s2_ids = {
        matched_id
        for _, matched_id in true_pairs
    }

    print(
        f"Unique true S2 entities needed: "
        f"{len(required_s2_ids):,}"
    )

    # =========================================================
    # 3. LOAD REQUIRED SOURCE 2 RECORDS
    # =========================================================

    source2 = load_required_source2(
        true_pairs
    )

    print(
        f"S2 records loaded for recall test: "
        f"{len(source2):,}"
    )

    loaded_s2_ids = set(
        source2["entity_id"]
    )

    missing_s2_ids = (
        required_s2_ids
        - loaded_s2_ids
    )

    if missing_s2_ids:

        print(
            "\nWARNING:"
        )

        print(
            f"Missing required S2 IDs: "
            f"{len(missing_s2_ids):,}"
        )

        print(
            "These pairs cannot be evaluated "
            "from the loaded S2 records."
        )

    else:

        print(
            "All required S2 records loaded successfully."
        )

    # =========================================================
    # 4. GENERATE ADDRESS CANDIDATES
    # =========================================================

    candidates = build_address_candidates(
        source1,
        source2,
    )

    print(
        f"\nGenerated address candidates: "
        f"{len(candidates):,}"
    )

    # =========================================================
    # 5. CALCULATE RECALL
    # =========================================================

    evaluable_true_pairs = {
        pair
        for pair in true_pairs
        if pair[1] in loaded_s2_ids
    }

    recovered = (
        evaluable_true_pairs
        .intersection(candidates)
    )

    missed = (
        evaluable_true_pairs
        - candidates
    )

    recall = (
        len(recovered)
        / len(evaluable_true_pairs)
        * 100
        if evaluable_true_pairs
        else 0
    )

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)

    print(
        f"Evaluable true pairs: "
        f"{len(evaluable_true_pairs):,}"
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
        f"Address recall:       "
        f"{recall:.2f}%"
    )

    # =========================================================
    # 6. SAMPLE RECOVERED PAIRS
    # =========================================================

    print("\n" + "-" * 70)
    print("SAMPLE RECOVERED PAIRS")
    print("-" * 70)

    for source1_id, source2_id in list(
        recovered
    )[:10]:

        s1_matches = source1[
            source1["entity_id"]
            == source1_id
        ]

        s2_matches = source2[
            source2["entity_id"]
            == source2_id
        ]

        if s1_matches.empty or s2_matches.empty:
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

    # =========================================================
    # 7. SAMPLE MISSED PAIRS
    # =========================================================

    print("\n" + "-" * 70)
    print("SAMPLE STILL-MISSED PAIRS")
    print("-" * 70)

    for source1_id, source2_id in list(
        missed
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