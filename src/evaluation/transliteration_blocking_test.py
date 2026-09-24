import re
import pandas as pd
import unicodedata
from collections import defaultdict

from src.preprocessing.normalize import normalize_text


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN_DIR = DATASET_ROOT + r"\dataset\train"


# ---------------------------------------------------------
# Basic script detection
# ---------------------------------------------------------

def contains_non_latin(text):
    if not text:
        return False

    for char in text:
        if char.isalpha() and ord(char) > 127:
            return True

    return False


def normalize_multilingual_name(value):
    """
    Keep the original Unicode characters but normalize
    spacing, punctuation and case.
    """

    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    value = re.sub(
        r"[\.,;:/\\|_\-]+",
        " ",
        value
    )

    value = re.sub(
        r"[\(\)\[\]\{\}]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ---------------------------------------------------------
# Character n-grams
# ---------------------------------------------------------

def character_ngrams(value, n=3):

    value = normalize_multilingual_name(value)

    if not value:
        return []

    # Remove spaces for cross-script character matching.
    value = value.replace(" ", "")

    if len(value) < n:
        return [value]

    return [
        value[i:i+n]
        for i in range(len(value) - n + 1)
    ]


# ---------------------------------------------------------
# Load Source 1
# ---------------------------------------------------------

def load_source1():

    path = TRAIN_DIR + r"\train_source1.tsv"

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=5000
    )

    df["name_norm"] = df[
        "business_name"
    ].apply(normalize_multilingual_name)

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# ---------------------------------------------------------
# Ground truth
# ---------------------------------------------------------

def load_ground_truth(sample_s1_ids):

    path = TRAIN_DIR + r"\train_ground_truth.tsv"

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    sample_ids = set(sample_s1_ids)

    gt = gt[
        gt["source1_entity_id"].isin(sample_ids)
    ]

    true_pairs = set()

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for matched_id in matched_ids.split(","):

            matched_id = matched_id.strip()

            # Only Source 2 for this experiment.
            if matched_id.startswith("S2-"):
                true_pairs.add(
                    (s1_id, matched_id)
                )

    return true_pairs


# ---------------------------------------------------------
# Load only required S2 rows
# ---------------------------------------------------------

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

    df = pd.concat(
        chunks,
        ignore_index=True
    )

    df["name_norm"] = df[
        "business_name"
    ].apply(normalize_multilingual_name)

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


# ---------------------------------------------------------
# Build multilingual character n-gram index
# ---------------------------------------------------------

def build_ngram_index(source2):

    index = defaultdict(set)

    for _, row in source2.iterrows():

        s2_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]

        if not name:
            continue

        grams = set(
            character_ngrams(
                name,
                n=3
            )
        )

        for gram in grams:

            index[
                (country, gram)
            ].add(s2_id)

    return index


# ---------------------------------------------------------
# Generate candidates
# ---------------------------------------------------------

def generate_candidates(source1, index):

    candidates = set()

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]

        if not name:
            continue

        grams = set(
            character_ngrams(
                name,
                n=3
            )
        )

        matched_ids = set()

        for gram in grams:

            key = (
                country,
                gram
            )

            matched_ids.update(
                index.get(key, set())
            )

        for s2_id in matched_ids:

            candidates.add(
                (s1_id, s2_id)
            )

    return candidates


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Loading Source 1 sample...")

    source1 = load_source1()

    print(
        f"Source 1 rows: {len(source1):,}"
    )

    # -----------------------------------------------------
    # Ground truth
    # -----------------------------------------------------

    true_pairs = load_ground_truth(
        source1["entity_id"]
    )

    print(
        f"True S1 -> S2 pairs: "
        f"{len(true_pairs):,}"
    )

    required_s2_ids = {
        s2_id
        for _, s2_id in true_pairs
    }

    # -----------------------------------------------------
    # Load required S2
    # -----------------------------------------------------

    print(
        "Loading required Source 2 records..."
    )

    source2 = load_required_source2(
        required_s2_ids
    )

    print(
        f"Source 2 rows: {len(source2):,}"
    )

    # -----------------------------------------------------
    # Build index
    # -----------------------------------------------------

    print(
        "\nBuilding multilingual "
        "character n-gram index..."
    )

    index = build_ngram_index(
        source2
    )

    print(
        f"Index keys: {len(index):,}"
    )

    # -----------------------------------------------------
    # Generate candidates
    # -----------------------------------------------------

    print(
        "\nGenerating transliteration/"
        "cross-script candidates..."
    )

    candidates = generate_candidates(
        source1,
        index
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    recovered = (
        true_pairs &
        candidates
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

    print("\n" + "=" * 60)
    print(
        "MULTILINGUAL NAME BLOCKING TEST"
    )
    print("=" * 60)

    print(
        f"Candidate pairs:       "
        f"{len(candidates):,}"
    )

    print(
        f"Average candidates/S1: "
        f"{len(candidates) / len(source1):.2f}"
    )

    print(
        f"Recovered pairs:       "
        f"{len(recovered):,}"
    )

    print(
        f"Missed pairs:          "
        f"{len(missed):,}"
    )

    print(
        f"Multilingual recall:   "
        f"{recall * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Show missed pairs
    # -----------------------------------------------------

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
    