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

    # Cache features once per entity. This avoids repeated normalization and
    # DataFrame lookups for every candidate pair in large challenge splits.
    name_frequency = Counter()
    source_profiles = {}
    for row in s2s3_df.itertuples(index=False):
        entity_id, business_name, business_address = row[:3]
        name = normalize_simple(business_name)
        address = normalize_simple(business_address)
        n_tokens = name_tokens(name)
        for token in n_tokens:
            name_frequency[token] += 1
        source_profiles[str(entity_id)] = (
            name, n_tokens, address_tokens(address), numeric_tokens(address)
        )

    s1_profiles = {}
    for row in s1_df.itertuples(index=False):
        entity_id, business_name, business_address = row[:3]
        name = normalize_simple(business_name)
        address = normalize_simple(business_address)
        s1_profiles[str(entity_id)] = (
            name, name_tokens(name), address_tokens(address), numeric_tokens(address)
        )

    output_rows = []

    before_count = 0
    after_count = 0

    for s1_id, candidate_string in candidates_df[
        ["source1_entity_id", "candidate_entity_ids"]
    ].itertuples(index=False, name=None):
        s1_id = str(s1_id)
        s1_profile = s1_profiles.get(s1_id)
        if s1_profile is None:
            continue
        name1, n1, a1, numeric_a1 = s1_profile
        candidate_ids = str(candidate_string or "").split(",")

        kept = []

        for candidate_id in candidate_ids:

            before_count += 1
            candidate_profile = source_profiles.get(candidate_id)
            if candidate_profile is None:
                continue
            name2, n2, a2, numeric_a2 = candidate_profile
            shared_name = n1 & n2
            shared_address = a1 & a2
            keep = bool(name1 and name2 and name1 == name2)
            keep = keep or bool(numeric_a1 & numeric_a2)
            keep = keep or len(shared_address) >= 2
            keep = keep or bool(shared_name and shared_address)
            keep = keep or any(
                name_frequency.get(token, 0) <= rare_name_threshold
                for token in shared_name
            )
            if keep:
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
