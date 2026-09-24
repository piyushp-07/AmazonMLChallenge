import re
import pandas as pd

from src.blocking.candidate_generation import (
    load_source,
    generate_candidates,
    TRAIN_DIR,
)


SAMPLE_SIZE = 5000


def short_name_tokens(value):
    """
    Extract alphabetic/alphanumeric name tokens,
    including 2-character tokens.
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
        and len(token) >= 2
    ]


def load_ground_truth(sample_s1_ids):

    gt = pd.read_csv(
        TRAIN_DIR + r"\train_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = gt[
        gt["source1_entity_id"].isin(
            set(sample_s1_ids)
        )
    ]

    true_pairs = set()

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for matched_id in matched_ids.split(","):

            matched_id = matched_id.strip()

            if matched_id.startswith("S2-"):
                true_pairs.add(
                    (s1_id, matched_id)
                )

    return true_pairs


def build_short_token_index(source2):

    index = {}

    for _, row in source2.iterrows():

        country = row["country_norm"]
        name = row["name_norm"]
        entity_id = row["entity_id"]

        if not country or not name:
            continue

        for token in short_name_tokens(name):

            key = (country, token)

            if key not in index:
                index[key] = set()

            index[key].add(entity_id)

    return index


def main():

    print("=" * 60)
    print("SHORT NAME TOKEN BLOCKING TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\nLoading Source 1...")

    source1 = load_source(
        TRAIN_DIR + r"\train_source1.tsv"
    ).head(SAMPLE_SIZE)

    print(
        f"Source 1 rows: {len(source1):,}"
    )

    print("\nLoading Source 2...")

    source2 = load_source(
        TRAIN_DIR + r"\train_source2.tsv"
    )

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------

    print("\nLoading ground truth...")

    true_pairs = load_ground_truth(
        source1["entity_id"]
    )

    print(
        f"True S1 -> S2 pairs: "
        f"{len(true_pairs):,}"
    )

    # --------------------------------------------------
    # Existing production blocker
    # --------------------------------------------------

    print(
        "\nRunning existing production blocker..."
    )

    existing_df = generate_candidates(
        source1,
        source2[
            source2["entity_id"].isin(
                {
                    s2
                    for _, s2 in true_pairs
                }
            )
        ]
    )

    existing_candidates = set(
        zip(
            existing_df[
                "source1_entity_id"
            ],
            existing_df[
                "matched_entity_id"
            ],
        )
    )

    existing_recovered = (
        true_pairs &
        existing_candidates
    )

    existing_missed = (
        true_pairs -
        existing_candidates
    )

    print(
        f"Existing candidates: "
        f"{len(existing_candidates):,}"
    )

    print(
        f"Existing recovered: "
        f"{len(existing_recovered):,}"
    )

    print(
        f"Existing missed: "
        f"{len(existing_missed):,}"
    )

    # --------------------------------------------------
    # Build short-token index
    # --------------------------------------------------

    print(
        "\nBuilding 2-character token index..."
    )

    token_index = build_short_token_index(
        source2
    )

    # --------------------------------------------------
    # Add short-token candidates
    # --------------------------------------------------

    short_candidates = set()

    for _, row in source1.iterrows():

        s1_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]

        if not country or not name:
            continue

        for token in short_name_tokens(name):

            matches = token_index.get(
                (country, token),
                set()
            )

            for s2_id in matches:

                short_candidates.add(
                    (s1_id, s2_id)
                )

    # --------------------------------------------------
    # Combined result
    # --------------------------------------------------

    combined_candidates = (
        existing_candidates |
        short_candidates
    )

    recovered = (
        true_pairs &
        combined_candidates
    )

    missed = (
        true_pairs -
        combined_candidates
    )

    added = (
        combined_candidates -
        existing_candidates
    )

    newly_recovered = (
        recovered -
        existing_recovered
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Existing candidates: "
        f"{len(existing_candidates):,}"
    )

    print(
        f"Short-token candidates: "
        f"{len(short_candidates):,}"
    )

    print(
        f"New candidates added: "
        f"{len(added):,}"
    )

    print(
        f"Combined candidates: "
        f"{len(combined_candidates):,}"
    )

    print(
        f"\nExisting recall: "
        f"{len(existing_recovered) / len(true_pairs) * 100:.2f}%"
    )

    print(
        f"Combined recall: "
        f"{len(recovered) / len(true_pairs) * 100:.2f}%"
    )

    print(
        f"Newly recovered pairs: "
        f"{len(newly_recovered):,}"
    )

    print(
        f"Remaining missed pairs: "
        f"{len(missed):,}"
    )

    print(
        f"Candidates/S1: "
        f"{len(combined_candidates) / len(source1):.2f}"
    )

    # --------------------------------------------------
    # Inspect remaining misses
    # --------------------------------------------------

    if missed:

        print("\nRemaining misses:")

        for pair in list(missed)[:30]:

            print(
                f"{pair[0]} -> {pair[1]}"
            )

    else:

        print(
            "\nAll true pairs recovered."
        )


if __name__ == "__main__":
    main()