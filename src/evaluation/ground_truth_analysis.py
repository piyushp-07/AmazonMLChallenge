from pathlib import Path
import pandas as pd
from collections import Counter


DATASET_ROOT = Path(
    r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
)

GROUND_TRUTH = (
    DATASET_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


def analyze_ground_truth():
    print("=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(f"\nTotal Source 1 entities: {len(df):,}")

    # ---------------------------------------------------------
    # Count number of matches for every Source 1 entity
    # ---------------------------------------------------------

    match_counts = []

    for value in df["matched_entity_ids"]:
        if not value.strip():
            match_counts.append(0)
        else:
            matches = [
                x.strip()
                for x in value.split(",")
                if x.strip()
            ]
            match_counts.append(len(matches))

    df["match_count"] = match_counts

    # ---------------------------------------------------------
    # Match-count distribution
    # ---------------------------------------------------------

    distribution = (
        df["match_count"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "-" * 80)
    print("MATCH COUNT DISTRIBUTION")
    print("-" * 80)

    for count, entities in distribution.items():
        percentage = entities / len(df) * 100

        print(
            f"{count:>3} matches : "
            f"{entities:>10,} entities "
            f"({percentage:>6.2f}%)"
        )

    # ---------------------------------------------------------
    # Singleton / zero / multi-match statistics
    # ---------------------------------------------------------

    zero_match = (df["match_count"] == 0).sum()
    singleton = (df["match_count"] == 1).sum()
    multi_match = (df["match_count"] > 1).sum()

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    print(f"Zero matches     : {zero_match:,}")
    print(f"One match        : {singleton:,}")
    print(f"Multiple matches : {multi_match:,}")

    # ---------------------------------------------------------
    # Total number of ground-truth links
    # ---------------------------------------------------------

    total_links = df["match_count"].sum()

    print(f"\nTotal ground-truth links: {total_links:,}")

    # ---------------------------------------------------------
    # S2 vs S3 match distribution
    # ---------------------------------------------------------

    s2_count = 0
    s3_count = 0
    other_count = 0

    for value in df["matched_entity_ids"]:

        if not value.strip():
            continue

        for entity_id in value.split(","):

            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                s2_count += 1

            elif entity_id.startswith("S3-"):
                s3_count += 1

            else:
                other_count += 1

    print("\n" + "-" * 80)
    print("MATCH SOURCE DISTRIBUTION")
    print("-" * 80)

    print(f"S2 matches : {s2_count:,}")
    print(f"S3 matches : {s3_count:,}")
    print(f"Other      : {other_count:,}")

    # ---------------------------------------------------------
    # Match count statistics
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("MATCH COUNT STATISTICS")
    print("-" * 80)

    print(df["match_count"].describe())

    # ---------------------------------------------------------
    # Top entities with most matches
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("ENTITIES WITH MOST MATCHES")
    print("-" * 80)

    top = df.nlargest(10, "match_count")[
        ["source1_entity_id", "matched_entity_ids", "match_count"]
    ]

    print(top.to_string(index=False))

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    print("\n" + "-" * 80)
    print("SANITY CHECKS")
    print("-" * 80)

    invalid_s2 = 0
    invalid_s3 = 0

    for value in df["matched_entity_ids"]:

        if not value.strip():
            continue

        for entity_id in value.split(","):

            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                pass
            elif entity_id.startswith("S3-"):
                pass
            else:
                if entity_id.startswith("S2"):
                    invalid_s2 += 1
                elif entity_id.startswith("S3"):
                    invalid_s3 += 1

    print(f"Potential malformed S2 IDs: {invalid_s2}")
    print(f"Potential malformed S3 IDs: {invalid_s3}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    analyze_ground_truth()