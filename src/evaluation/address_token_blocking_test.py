import re
import pandas as pd
from collections import defaultdict

from src.preprocessing.normalize import normalize_address


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN_DIR = DATASET_ROOT + r"\dataset\train"


def load_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    df["address_norm"] = df["business_address"].apply(normalize_address)
    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def load_ground_truth(sample_s1_ids):
    gt_path = TRAIN_DIR + r"\train_ground_truth.tsv"

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    sample_s1_ids = set(sample_s1_ids)

    gt = gt[gt["source1_entity_id"].isin(sample_s1_ids)]

    true_pairs = set()

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for matched_id in matched_ids.split(","):
            matched_id = matched_id.strip()

            # IMPORTANT:
            # This experiment is only for Source 2.
            if matched_id.startswith("S2-"):
                true_pairs.add((s1_id, matched_id))

    return true_pairs


def load_required_source2(required_ids):
    path = TRAIN_DIR + r"\train_source2.tsv"

    required_ids = set(required_ids)

    chunks = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100000
    ):
        matched = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if not matched.empty:
            chunks.append(matched)

        if sum(len(x) for x in chunks) >= len(required_ids):
            break

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    df["address_norm"] = df["business_address"].apply(normalize_address)

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def extract_address_tokens(address):
    """
    Extract useful address tokens.

    Examples:
        '57B, Purna Das Road, Kolkata'
        ->
        ['57b', 'purna', 'das', 'road', 'kolkata']

    Very short/common tokens are removed later.
    """

    if not address:
        return []

    address = address.lower()

    # Keep alphanumeric words.
    tokens = re.findall(r"[a-z0-9]+", address)

    return tokens


def build_address_token_index(source2):
    """
    Build:

        (country, address_token) -> S2 IDs

    We ignore very common address words because they would
    generate huge candidate blocks.
    """

    token_to_ids = defaultdict(set)

    # Common address words that are not useful for blocking.
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
        "place",
        "pl",
        "park",
        "city",
        "town",
        "state",
        "county",
        "district",
        "india",
        "usa",
        "us",
        "the",
        "and",
        "near",
        "north",
        "south",
        "east",
        "west",
    }

    for _, row in source2.iterrows():

        s2_id = row["entity_id"]
        country = row["country_norm"]

        tokens = extract_address_tokens(
            row["address_norm"]
        )

        for token in set(tokens):

            if token in stopwords:
                continue

            # Ignore extremely short alphabetic tokens.
            if token.isalpha() and len(token) < 3:
                continue

            # Keep numbers and alphanumeric house numbers.
            token_to_ids[(country, token)].add(s2_id)

    return token_to_ids


def generate_candidates(source1, token_index):
    """
    Generate candidates when S1 and S2 share at least
    one useful address token within the same country.
    """

    candidates = set()

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]
        country = row["country_norm"]

        tokens = extract_address_tokens(
            row["address_norm"]
        )

        for token in set(tokens):

            key = (country, token)

            for s2_id in token_index.get(key, set()):

                candidates.add(
                    (s1_id, s2_id)
                )

    return candidates


def main():

    print("Loading Source 1 sample...")

    source1 = pd.read_csv(
        TRAIN_DIR + r"\train_source1.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=5000
    )

    source1["address_norm"] = source1[
        "business_address"
    ].apply(normalize_address)

    source1["country_norm"] = (
        source1["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(f"Source 1 rows: {len(source1):,}")

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------

    true_pairs = load_ground_truth(
        source1["entity_id"]
    )

    print(
        f"True S1 -> S2 pairs: {len(true_pairs):,}"
    )

    required_s2_ids = {
        s2_id
        for _, s2_id in true_pairs
    }

    # --------------------------------------------------
    # Load required S2 rows
    # --------------------------------------------------

    source2 = load_required_source2(
        required_s2_ids
    )

    print(
        f"Source 2 rows: {len(source2):,}"
    )

    # --------------------------------------------------
    # Build index
    # --------------------------------------------------

    print(
        "\nBuilding address-token index..."
    )

    token_index = build_address_token_index(
        source2
    )

    print(
        f"Address-token keys: {len(token_index):,}"
    )

    # --------------------------------------------------
    # Generate candidates
    # --------------------------------------------------

    print(
        "\nGenerating address-token candidates..."
    )

    candidates = generate_candidates(
        source1,
        token_index
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    recovered = true_pairs.intersection(
        candidates
    )

    missed = true_pairs - candidates

    recall = (
        len(recovered) / len(true_pairs)
        if true_pairs
        else 0
    )

    print("\n" + "=" * 60)
    print("ADDRESS TOKEN BLOCKING TEST")
    print("=" * 60)

    print(
        f"Candidate pairs:       {len(candidates):,}"
    )

    print(
        f"Average candidates/S1: "
        f"{len(candidates) / len(source1):.2f}"
    )

    print(
        f"Recovered pairs:       {len(recovered):,}"
    )

    print(
        f"Missed pairs:          {len(missed):,}"
    )

    print(
        f"Address-token recall:  "
        f"{recall * 100:.2f}%"
    )

    # --------------------------------------------------
    # Show missed pairs
    # --------------------------------------------------

    if missed:

        print("\n" + "=" * 60)
        print("MISSED TRUE PAIRS")
        print("=" * 60)

        s1_lookup = source1.set_index(
            "entity_id"
        )

        s2_lookup = source2.set_index(
            "entity_id"
        )

        for s1_id, s2_id in list(missed)[:30]:

            s1 = s1_lookup.loc[s1_id]
            s2 = s2_lookup.loc[s2_id]

            print("\nS1:", s1_id)
            print(
                "  Name:",
                s1["business_name"]
            )
            print(
                "  Address:",
                s1["business_address"]
            )

            print("S2:", s2_id)
            print(
                "  Name:",
                s2["business_name"]
            )
            print(
                "  Address:",
                s2["business_address"]
            )


if __name__ == "__main__":
    main()