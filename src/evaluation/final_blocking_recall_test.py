import pandas as pd

from src.blocking.candidate_generation import (
    load_source,
    generate_candidates,
    TRAIN_DIR,
)


SAMPLE_SIZE = 5000


def load_ground_truth(sample_s1_ids):
    """
    Load S1 -> S2 ground-truth pairs for the selected S1 sample.
    """

    gt_path = TRAIN_DIR + r"\train_ground_truth.tsv"

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    sample_s1_ids = set(sample_s1_ids)

    gt = gt[
        gt["source1_entity_id"].isin(sample_s1_ids)
    ]

    true_pairs = set()

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for matched_id in matched_ids.split(","):

            matched_id = matched_id.strip()

            # Keep ONLY Source 2 matches
            if matched_id.startswith("S2-"):
                true_pairs.add(
                    (s1_id, matched_id)
                )

    return true_pairs


def load_required_source2(required_ids):
    """
    Load Source 2 using the production load_source()
    function, then keep only the S2 records required
    by the ground truth.
    """

    path = TRAIN_DIR + r"\train_source2.tsv"

    print("Reading Source 2 with production loader...")

    # IMPORTANT:
    # This applies exactly the preprocessing expected
    # by generate_candidates().
    source2 = load_source(path)

    required_ids = set(required_ids)

    source2 = source2[
        source2["entity_id"].isin(required_ids)
    ].copy()

    return source2


def main():

    print("=" * 60)
    print("FINAL PRODUCTION BLOCKING RECALL TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Load Source 1
    # --------------------------------------------------

    print("\nLoading Source 1 sample...")

    source1 = load_source(
        TRAIN_DIR + r"\train_source1.tsv"
    )

    source1 = source1.head(
        SAMPLE_SIZE
    )

    print(
        f"Source 1 rows: "
        f"{len(source1):,}"
    )

    # --------------------------------------------------
    # Ground truth
    # --------------------------------------------------

    print("\nLoading Source 2 ground truth...")

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

    print(
        f"Unique true S2 entities: "
        f"{len(required_s2_ids):,}"
    )

    # --------------------------------------------------
    # Load required Source 2 records
    # --------------------------------------------------

    print(
        "\nLoading required Source 2 records..."
    )

    source2 = load_required_source2(
        required_s2_ids
    )

    print(
        f"Source 2 rows loaded: "
        f"{len(source2):,}"
    )

    # --------------------------------------------------
    # Generate candidates using production blocker
    # --------------------------------------------------

    print(
        "\nRunning production "
        "candidate_generation.py..."
    )

    candidates_df = generate_candidates(
        source1,
        source2
    )

    candidates = set(
        zip(
            candidates_df[
                "source1_entity_id"
            ],
            candidates_df[
                "matched_entity_id"
            ],
        )
    )

    print(
        f"Generated candidates: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------
    # Recall calculation
    # --------------------------------------------------

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
    print("RESULT")
    print("=" * 60)

    print(
        f"True pairs:       "
        f"{len(true_pairs):,}"
    )

    print(
        f"Recovered pairs:  "
        f"{len(recovered):,}"
    )

    print(
        f"Missed pairs:     "
        f"{len(missed):,}"
    )

    print(
        f"Blocking recall:  "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Candidates/S1:    "
        f"{len(candidates) / len(source1):.2f}"
    )

    # --------------------------------------------------
    # Show missed pairs
    # --------------------------------------------------

    if missed:

        print("\n" + "=" * 60)
        print("MISSED TRUE PAIRS")
        print("=" * 60)

        for s1_id, s2_id in list(missed)[:30]:

            print(
                f"{s1_id} -> {s2_id}"
            )

    else:

        print("\nNo true pairs were missed.")


if __name__ == "__main__":
    main()