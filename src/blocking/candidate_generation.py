import re
import pandas as pd
from collections import defaultdict

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
    normalize_country,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = (
    r"C:\Users\shiva\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource"
)

TRAIN_DIR = DATASET_ROOT + r"\dataset\train"


# ============================================================
# LOAD + NORMALIZE
# ============================================================

def load_source(path):
    """
    Load a source TSV and create normalized fields.
    """

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    df["name_norm"] = df[
        "business_name"
    ].apply(normalize_business_name)

    df["address_norm"] = df[
        "business_address"
    ].apply(normalize_address)

    df["country_norm"] = df[
        "country"
    ].apply(normalize_country)

    return df


# ============================================================
# NAME HELPERS
# ============================================================

def name_prefix(value, length):
    """
    Return the first `length` characters
    of a normalized name.
    """

    if not value:
        return ""

    return value[:length]


def name_tokens(value):
    """
    Extract useful Latin/number tokens from
    a normalized business name.
    """

    if not value:
        return []

    tokens = re.findall(
        r"[a-z0-9]+",
        value.lower()
    )

    stopwords = {
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

    return [
        token
        for token in set(tokens)
        if token not in stopwords
        and len(token) >= 3
    ]


# ============================================================
# ADDRESS HELPERS
# ============================================================

def address_tokens(value):
    """
    Extract useful alphanumeric address tokens.

    We deliberately ignore very short tokens because
    single-character address tokens create huge blocks.
    """

    if not value:
        return []

    tokens = re.findall(
        r"[a-z0-9]+",
        value.lower()
    )

    return [
        token
        for token in set(tokens)
        if len(token) >= 3
    ]


def address_prefix(value, length):
    """
    Return the first `length` characters of
    a normalized address.
    """

    if not value:
        return ""

    return value[:length]


# ============================================================
# INDEX BUILDING
# ============================================================

def build_source_index(source_df):
    """
    Build blocking indexes for Source 2.

    Blocking rules:

    1. Exact normalized name + country
    2. 4-character name prefix + country
    3. 6-character name prefix + country
    4. Useful name token + country
    5. Address token + country
    6. Address prefix + country
    """

    indexes = {
        "exact_name": defaultdict(set),
        "prefix_4": defaultdict(set),
        "prefix_6": defaultdict(set),
        "token": defaultdict(set),

        "address_token": defaultdict(set),
        "address_prefix_4": defaultdict(set),
        "address_prefix_6": defaultdict(set),
    }

    for _, row in source_df.iterrows():

        entity_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]
        address = row["address_norm"]

        # ----------------------------------------------------
        # NAME BLOCKING
        # ----------------------------------------------------

        if country and name:

            # Exact name
            indexes["exact_name"][
                (country, name)
            ].add(entity_id)

            # 4-char prefix
            prefix4 = name_prefix(
                name,
                4
            )

            if prefix4:
                indexes["prefix_4"][
                    (country, prefix4)
                ].add(entity_id)

            # 6-char prefix
            prefix6 = name_prefix(
                name,
                6
            )

            if prefix6:
                indexes["prefix_6"][
                    (country, prefix6)
                ].add(entity_id)

            # Useful name tokens
            for token in name_tokens(name):

                indexes["token"][
                    (country, token)
                ].add(entity_id)

        # ----------------------------------------------------
        # ADDRESS BLOCKING
        # ----------------------------------------------------

        if country and address:

            # Address prefix
            prefix4 = address_prefix(
                address,
                4
            )

            if prefix4:
                indexes["address_prefix_4"][
                    (country, prefix4)
                ].add(entity_id)

            prefix6 = address_prefix(
                address,
                6
            )

            if prefix6:
                indexes["address_prefix_6"][
                    (country, prefix6)
                ].add(entity_id)

            # Address tokens
            for token in address_tokens(address):

                indexes["address_token"][
                    (country, token)
                ].add(entity_id)

    return indexes


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(source1, source2):
    """
    Generate Source1 -> Source2 candidate pairs.

    Name blocking:
        1. Exact normalized name
        2. 4-char prefix
        3. 6-char prefix
        4. Useful name token

    Address blocking:
        5. Address token
        6. Address prefix

    All blocking rules are country-aware.
    """

    source2_index = build_source_index(
        source2
    )

    candidates = set()

    # --------------------------------------------------------
    # Process Source 1
    # --------------------------------------------------------

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]
        address = row["address_norm"]

        if not country:
            continue

        # ====================================================
        # NAME RULE 1: Exact normalized name
        # ====================================================

        if name:

            candidates.update(
                (s1_id, s2_id)
                for s2_id in source2_index[
                    "exact_name"
                ].get(
                    (country, name),
                    set()
                )
            )

        # ====================================================
        # NAME RULE 2: 4-character prefix
        # ====================================================

        if name:

            prefix4 = name_prefix(
                name,
                4
            )

            if prefix4:

                candidates.update(
                    (s1_id, s2_id)
                    for s2_id in source2_index[
                        "prefix_4"
                    ].get(
                        (country, prefix4),
                        set()
                    )
                )

        # ====================================================
        # NAME RULE 3: 6-character prefix
        # ====================================================

        if name:

            prefix6 = name_prefix(
                name,
                6
            )

            if prefix6:

                candidates.update(
                    (s1_id, s2_id)
                    for s2_id in source2_index[
                        "prefix_6"
                    ].get(
                        (country, prefix6),
                        set()
                    )
                )

        # ====================================================
        # NAME RULE 4: Useful name tokens
        # ====================================================

        if name:

            for token in name_tokens(name):

                candidates.update(
                    (s1_id, s2_id)
                    for s2_id in source2_index[
                        "token"
                    ].get(
                        (country, token),
                        set()
                    )
                )

        # ====================================================
        # ADDRESS RULE 5: Address tokens
        # ====================================================

        if address:

            for token in address_tokens(address):

                candidates.update(
                    (s1_id, s2_id)
                    for s2_id in source2_index[
                        "address_token"
                    ].get(
                        (country, token),
                        set()
                    )
                )

        # ====================================================
        # ADDRESS RULE 6: Address prefixes
        # ====================================================

        if address:

            prefix4 = address_prefix(
                address,
                4
            )

            if prefix4:

                candidates.update(
                    (s1_id, s2_id)
                    for s2_id in source2_index[
                        "address_prefix_4"
                    ].get(
                        (country, prefix4),
                        set()
                    )
                )

            prefix6 = address_prefix(
                address,
                6
            )

            if prefix6:

                candidates.update(
                    (s1_id, s2_id)
                    for s2_id in source2_index[
                        "address_prefix_6"
                    ].get(
                        (country, prefix6),
                        set()
                    )
                )

    return pd.DataFrame(
        list(candidates),
        columns=[
            "source1_entity_id",
            "matched_entity_id",
        ],
    )


# ============================================================
# SMALL-SAMPLE TEST
# ============================================================

def main():

    print("Loading sample data...")

    source1 = load_source(
        TRAIN_DIR + r"\train_source1.tsv"
    )

    source2 = load_source(
        TRAIN_DIR + r"\train_source2.tsv"
    )

    source1 = source1.head(5000)
    source2 = source2.head(10000)

    print(
        f"Source 1 rows: {len(source1):,}"
    )

    print(
        f"Source 2 rows: {len(source2):,}"
    )

    print(
        "\nGenerating candidates..."
    )

    candidates = generate_candidates(
        source1,
        source2
    )

    print(
        f"Candidate pairs: "
        f"{len(candidates):,}"
    )

    if len(source1) > 0:

        print(
            "Average candidates/S1: "
            f"{len(candidates) / len(source1):.2f}"
        )

    print("\nSample candidates:")

    print(
        candidates.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()