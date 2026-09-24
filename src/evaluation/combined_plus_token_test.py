import os
import re
import pandas as pd

from src.preprocessing.normalize import normalize_business_name, normalize_address


DATASET_ROOT = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN_DIR = os.path.join(DATASET_ROOT, "dataset", "train")


def load_ground_truth(s1_ids):
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

    gt = pd.read_csv(
        gt_path,
        sep="\t",
        dtype=str
    )

    gt = gt[gt["source1_entity_id"].isin(s1_ids)]

    true_pairs = set()

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched = str(row["matched_entity_ids"])

        if matched == "nan" or not matched.strip():
            continue

        for matched_id in matched.split(","):
            matched_id = matched_id.strip()

            # Only Source 2
            if matched_id.startswith("S2-"):
                true_pairs.add((s1_id, matched_id))

    return true_pairs


def load_required_source2(required_ids):
    path = os.path.join(TRAIN_DIR, "train_source2.tsv")

    chunks = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100_000
    ):
        found = chunk[chunk["entity_id"].isin(required_ids)]

        if not found.empty:
            chunks.append(found)

        if sum(len(x) for x in chunks) >= len(required_ids):
            break

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def normalize_dataframe(df):
    df = df.copy()

    df["name_norm"] = df["business_name"].apply(
        normalize_business_name
    )

    df["address_norm"] = df["business_address"].apply(
        normalize_address
    )

    df["country_norm"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def name_prefix(value, length):
    if not value:
        return ""
    return value[:length]


def build_current_name_candidates(source1, source2):
    candidates = set()

    s2_exact = {}
    s2_prefix4 = {}
    s2_prefix6 = {}

    for _, row in source2.iterrows():
        country = row["country_norm"]
        name = row["name_norm"]
        entity_id = row["entity_id"]

        if not country or not name:
            continue

        exact_key = (country, name)
        prefix4_key = (country, name_prefix(name, 4))
        prefix6_key = (country, name_prefix(name, 6))

        s2_exact.setdefault(exact_key, []).append(entity_id)
        s2_prefix4.setdefault(prefix4_key, []).append(entity_id)
        s2_prefix6.setdefault(prefix6_key, []).append(entity_id)

    for _, row in source1.iterrows():
        s1_id = row["entity_id"]
        country = row["country_norm"]
        name = row["name_norm"]

        if not country or not name:
            continue

        exact_key = (country, name)
        prefix4_key = (country, name_prefix(name, 4))
        prefix6_key = (country, name_prefix(name, 6))

        for s2_id in s2_exact.get(exact_key, []):
            candidates.add((s1_id, s2_id))

        for s2_id in s2_prefix4.get(prefix4_key, []):
            candidates.add((s1_id, s2_id))

        for s2_id in s2_prefix6.get(prefix6_key, []):
            candidates.add((s1_id, s2_id))

    return candidates


def build_address_candidates(source1, source2):
    candidates = set()

    s2_address_tokens = {}

    for _, row in source2.iterrows():
        address = row["address_norm"]

        if not address:
            continue

        tokens = set(re.findall(r"[a-z0-9]+", address))

        for token in tokens:
            if len(token) < 2:
                continue

            key = (row["country_norm"], token)

            s2_address_tokens.setdefault(key, set()).add(
                row["entity_id"]
            )

    for _, row in source1.iterrows():
        address = row["address_norm"]

        if not address:
            continue

        tokens = set(re.findall(r"[a-z0-9]+", address))

        if not tokens:
            continue

        candidate_counts = {}

        for token in tokens:
            if len(token) < 2:
                continue

            key = (row["country_norm"], token)

            for s2_id in s2_address_tokens.get(key, set()):
                candidate_counts[s2_id] = (
                    candidate_counts.get(s2_id, 0) + 1
                )

        # Require at least 2 shared address tokens.
        for s2_id, count in candidate_counts.items():
            if count >= 2:
                candidates.add((row["entity_id"], s2_id))

    return candidates


def build_token_candidates(source1, source2):
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

    index = {}

    for _, row in source2.iterrows():
        name = row["name_norm"]

        if not name:
            continue

        tokens = set(re.findall(r"[a-z0-9]+", name))

        informative_tokens = {
            token
            for token in tokens
            if len(token) >= 3 and token not in stopwords
        }

        for token in informative_tokens:
            key = (row["country_norm"], token)

            index.setdefault(key, set()).add(
                row["entity_id"]
            )

    candidates = set()

    for _, row in source1.iterrows():
        name = row["name_norm"]

        if not name:
            continue

        tokens = set(re.findall(r"[a-z0-9]+", name))

        informative_tokens = {
            token
            for token in tokens
            if len(token) >= 3 and token not in stopwords
        }

        for token in informative_tokens:
            key = (row["country_norm"], token)

            for s2_id in index.get(key, set()):
                candidates.add(
                    (row["entity_id"], s2_id)
                )

    return candidates


def evaluate(name, candidates, true_pairs):
    recovered = candidates & true_pairs
    missed = true_pairs - candidates

    recall = (
        len(recovered) / len(true_pairs)
        if true_pairs
        else 0
    )

    print(f"{name}")
    print("-" * 70)
    print(f"Candidates:       {len(candidates):,}")
    print(f"Recovered pairs:  {len(recovered):,}")
    print(f"Missed pairs:     {len(missed):,}")
    print(f"Recall:           {recall:.2%}")
    print()

    return recovered, missed


def main():

    print("=" * 70)
    print("COMBINED + TOKEN BLOCKING TEST")
    print("=" * 70)

    source1_path = os.path.join(
        TRAIN_DIR,
        "train_source1.tsv"
    )

    source1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        nrows=5000
    )

    source1 = normalize_dataframe(source1)

    true_pairs = load_ground_truth(
        set(source1["entity_id"])
    )

    required_s2_ids = {
        s2_id
        for _, s2_id in true_pairs
    }

    source2 = load_required_source2(required_s2_ids)
    source2 = normalize_dataframe(source2)

    print(f"\nSource 1 rows: {len(source1):,}")
    print(f"Source 2 rows: {len(source2):,}")
    print(f"True S1 -> S2 pairs: {len(true_pairs):,}")

    print("\nGenerating current name candidates...")
    name_candidates = build_current_name_candidates(
        source1,
        source2
    )

    print("Generating address candidates...")
    address_candidates = build_address_candidates(
        source1,
        source2
    )

    print("Generating token candidates...")
    token_candidates = build_token_candidates(
        source1,
        source2
    )

    current_combined = (
        name_candidates |
        address_candidates
    )

    combined_plus_token = (
        current_combined |
        token_candidates
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    current_recovered, current_missed = evaluate(
        "CURRENT NAME + ADDRESS",
        current_combined,
        true_pairs
    )

    token_recovered, token_missed = evaluate(
        "TOKEN ONLY",
        token_candidates,
        true_pairs
    )

    union_recovered, union_missed = evaluate(
        "CURRENT + TOKEN",
        combined_plus_token,
        true_pairs
    )

    newly_recovered = (
        union_recovered - current_recovered
    )

    added_candidates = (
        combined_plus_token - current_combined
    )

    print("=" * 70)
    print("TOKEN'S INCREMENTAL CONTRIBUTION")
    print("=" * 70)

    print(
        f"Currently missed by combined: "
        f"{len(current_missed):,}"
    )

    print(
        f"Recovered by adding token: "
        f"{len(newly_recovered):,}"
    )

    print(
        f"Still missed after adding token: "
        f"{len(union_missed):,}"
    )

    print(
        f"New candidate pairs added: "
        f"{len(added_candidates):,}"
    )

    print(
        f"Final candidates: "
        f"{len(combined_plus_token):,}"
    )

    print(
        f"Final average candidates/S1: "
        f"{len(combined_plus_token) / len(source1):.2f}"
    )

    if newly_recovered:
        print("\nNewly recovered true pairs:")

        for pair in sorted(newly_recovered):
            print(
                f"  {pair[0]} -> {pair[1]}"
            )

    if union_missed:
        print("\nStill missed true pairs:")

        for pair in sorted(union_missed):
            print(
                f"  {pair[0]} -> {pair[1]}"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()