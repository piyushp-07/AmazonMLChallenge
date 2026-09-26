import re
from collections import Counter
import pandas as pd


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


def normalize_simple(value):
    if value is None:
        return ""

    value = str(value).strip().lower()

    value = re.sub(
        r"[\.,;:/\\|_\-]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def name_tokens(value):
    value = normalize_simple(value)

    if not value:
        return set()

    tokens = re.findall(
        r"[a-z0-9]+",
        value,
    )

    return {
        token
        for token in tokens
        if len(token) >= 3
        and token not in NAME_STOPWORDS
    }


def address_tokens(value):
    value = normalize_simple(value)

    if not value:
        return set()

    tokens = re.findall(
        r"[a-z0-9]+",
        value,
    )

    return {
        token
        for token in tokens
        if len(token) >= 3
    }


def numeric_tokens(value):
    value = normalize_simple(value)

    if not value:
        return set()

    tokens = re.findall(
        r"[a-z0-9]+",
        value,
    )

    return {
        token
        for token in tokens
        if any(ch.isdigit() for ch in token)
        and len(token) >= 3
    }


def build_candidate_filter_indexes(source2s3_df):
    """
    Build cheap frequency indexes for Stage-2 filtering.
    """

    name_frequency = Counter()

    for value in source2s3_df["business_name"]:
        tokens = name_tokens(value)

        for token in tokens:
            name_frequency[token] += 1

    return {
        "name_frequency": name_frequency,
    }


def keep_candidate(
    s1_row,
    candidate_row,
    indexes,
    rare_name_threshold=100,
):
    """
    Cheap, conservative candidate filter.

    Returns True when there is enough inexpensive
    evidence to send the pair to expensive ML features.
    """

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

    # --------------------------------------------------
    # 1. Exact normalized name
    # --------------------------------------------------

    if name1 and name2 and name1 == name2:
        return True

    # --------------------------------------------------
    # Cheap token evidence
    # --------------------------------------------------

    n1 = name_tokens(name1)
    n2 = name_tokens(name2)

    a1 = address_tokens(address1)
    a2 = address_tokens(address2)

    numeric_a1 = numeric_tokens(address1)
    numeric_a2 = numeric_tokens(address2)

    shared_name = n1 & n2
    shared_address = a1 & a2
    shared_numeric_address = numeric_a1 & numeric_a2

    # --------------------------------------------------
    # 2. Strong address evidence
    #
    # Address is highly available in the data.
    # --------------------------------------------------

    if shared_numeric_address:
        return True

    if len(shared_address) >= 2:
        return True

    # --------------------------------------------------
    # 3. Name + address evidence
    # --------------------------------------------------

    if shared_name and shared_address:
        return True

    # --------------------------------------------------
    # 4. Rare name evidence
    #
    # Preserve name-only candidates when the shared
    # business token is sufficiently distinctive.
    # --------------------------------------------------

    for token in shared_name:
        if indexes["name_frequency"].get(token, 0) <= rare_name_threshold:
            return True

    return False


def cheap_filter_candidates(
    s1_df,
    candidates_df,
    s2s3_df,
    rare_name_threshold=100,
):
    """
    Apply Stage-2 cheap filtering.

    Input:
        candidates_df:
            source1_entity_id
            candidate_entity_ids

    Output:
        Same format, but with inexpensive weak
        candidates removed.
    """

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

        s1_id = row["source1_entity_id"]

        if s1_id not in source1_index.index:
            continue

        s1_row = source1_index.loc[s1_id]

        candidate_ids = [
            c.strip()
            for c in str(
                row.get("candidate_entity_ids", "")
            ).split(",")
            if c.strip()
        ]

        kept = []

        for candidate_id in candidate_ids:

            if candidate_id not in candidate_index.index:
                continue

            before_count += 1

            candidate_row = candidate_index.loc[
                candidate_id
            ]

            if keep_candidate(
                s1_row,
                candidate_row,
                indexes,
                rare_name_threshold,
            ):
                kept.append(candidate_id)
                after_count += 1

        output_rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(
                    sorted(set(kept))
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
        f"Stage-2 candidates before: {before_count:,}"
    )

    print(
        f"Stage-2 candidates after:  {after_count:,}"
    )

    if before_count:
        print(
            "Retention:",
            f"{after_count / before_count * 100:.2f}%"
        )

    return result