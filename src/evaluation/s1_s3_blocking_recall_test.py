import pandas as pd

from src.blocking.candidate_generation import (
    load_source,
    generate_candidates,
)


BASE = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN = BASE + r"\dataset\train"


S1_PATH = TRAIN + r"\train_source1.tsv"
S3_PATH = TRAIN + r"\train_source3.tsv"
GT_PATH = TRAIN + r"\train_ground_truth.tsv"


SAMPLE_SIZE = 20000


def load_ground_truth(path, sample_ids):
    gt = pd.read_csv(path, sep="\t", dtype=str).fillna("")

    gt = gt[gt["source1_entity_id"].isin(sample_ids)]

    pairs = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if not matched:
            continue

        for s3_id in matched.split(","):
            s3_id = s3_id.strip()

            if s3_id.startswith("S3-"):
                pairs.append((s1_id, s3_id))

    return set(pairs)


def main():
    print("=" * 60)
    print("S1 -> S3 BLOCKING RECALL TEST")
    print("=" * 60)

    print("\nLoading Source 1...")

    source1 = load_source(S1_PATH)
    source1 = source1.head(SAMPLE_SIZE).copy()

    print(f"Source 1 rows: {len(source1):,}")

    sample_ids = set(source1["entity_id"])

    print("\nLoading ground truth...")

    true_pairs = load_ground_truth(GT_PATH, sample_ids)

    print(f"True S1 -> S3 pairs: {len(true_pairs):,}")

    true_s3_ids = {s3_id for _, s3_id in true_pairs}

    print(f"Unique true S3 IDs: {len(true_s3_ids):,}")

    print("\nLoading required Source 3 records...")

    source3 = load_source(S3_PATH)

    source3 = source3[
        source3["entity_id"].isin(true_s3_ids)
    ].copy()

    print(f"Source 3 rows loaded: {len(source3):,}")

    print("\nRunning production blocker...")

    candidates = generate_candidates(source1, source3)

    candidate_pairs = {
        (row["source1_entity_id"], row["matched_entity_id"])
        for _, row in candidates.iterrows()
    }

    recovered = true_pairs & candidate_pairs
    missed = true_pairs - candidate_pairs

    recall = (
        len(recovered) / len(true_pairs)
        if true_pairs
        else 1.0
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"Sample S1: {len(source1):,}")
    print(f"True S1 -> S3 pairs: {len(true_pairs):,}")
    print(f"Candidates: {len(candidates):,}")
    print(f"Recovered: {len(recovered):,}")
    print(f"Missed: {len(missed):,}")
    print(f"Blocking recall: {recall:.4%}")

    print(
        f"Candidates/S1: "
        f"{len(candidates) / len(source1):.2f}"
    )

    if missed:
        print("\nMissed pairs:")

        for s1_id, s3_id in sorted(missed)[:20]:
            print(f"  {s1_id} -> {s3_id}")
    else:
        print("\nAll true pairs recovered.")


if __name__ == "__main__":
    main()
