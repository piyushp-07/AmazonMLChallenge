import pandas as pd

from src.blocking.candidate_generation import (
    load_source,
    TRAIN_DIR,
)


MISSED_PAIRS = [
    ("S1-903445032", "S2-689432168"),
    ("S1-163077720", "S2-906538643"),
]


def main():

    print("=" * 70)
    print("INSPECTING PRODUCTION BLOCKING MISSES")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Source 1
    # ------------------------------------------------------

    print("\nLoading Source 1...")

    source1 = load_source(
        TRAIN_DIR + r"\train_source1.tsv"
    )

    # ------------------------------------------------------
    # Load Source 2
    # ------------------------------------------------------

    print("Loading Source 2...")

    source2 = load_source(
        TRAIN_DIR + r"\train_source2.tsv"
    )

    # ------------------------------------------------------
    # Inspect each missed pair
    # ------------------------------------------------------

    for s1_id, s2_id in MISSED_PAIRS:

        s1 = source1[
            source1["entity_id"] == s1_id
        ]

        s2 = source2[
            source2["entity_id"] == s2_id
        ]

        print("\n" + "=" * 70)
        print(f"MISSED PAIR")
        print("=" * 70)

        print(f"S1 ID: {s1_id}")
        print(f"S2 ID: {s2_id}")

        if s1.empty:
            print("\nS1 record NOT FOUND")
            continue

        if s2.empty:
            print("\nS2 record NOT FOUND")
            continue

        s1 = s1.iloc[0]
        s2 = s2.iloc[0]

        print("\n--- SOURCE 1 ---")

        print(
            f"Entity ID : {s1['entity_id']}"
        )

        print(
            f"Name      : {s1['business_name']}"
        )

        print(
            f"Name norm  : {s1['name_norm']}"
        )

        print(
            f"Address   : {s1['business_address']}"
        )

        print(
            f"Addr norm : {s1['address_norm']}"
        )

        print(
            f"Country   : {s1['country']}"
        )

        print(
            f"Country norm : {s1['country_norm']}"
        )

        print("\n--- SOURCE 2 ---")

        print(
            f"Entity ID : {s2['entity_id']}"
        )

        print(
            f"Name      : {s2['business_name']}"
        )

        print(
            f"Name norm  : {s2['name_norm']}"
        )

        print(
            f"Address   : {s2['business_address']}"
        )

        print(
            f"Addr norm : {s2['address_norm']}"
        )

        print(
            f"Country   : {s2['country']}"
        )

        print(
            f"Country norm : {s2['country_norm']}"
        )

        # --------------------------------------------------
        # Compare simple blocking keys
        # --------------------------------------------------

        print("\n--- KEY COMPARISON ---")

        s1_name = s1["name_norm"]
        s2_name = s2["name_norm"]

        s1_addr = s1["address_norm"]
        s2_addr = s2["address_norm"]

        print(
            f"Name exact match      : "
            f"{s1_name == s2_name}"
        )

        print(
            f"Name prefix 4 match   : "
            f"{s1_name[:4] == s2_name[:4]}"
        )

        print(
            f"Name prefix 6 match   : "
            f"{s1_name[:6] == s2_name[:6]}"
        )

        print(
            f"Address exact match   : "
            f"{s1_addr == s2_addr}"
        )

        print(
            f"Address prefix 4     : "
            f"{s1_addr[:4] == s2_addr[:4]}"
        )

        print(
            f"Address prefix 6     : "
            f"{s1_addr[:6] == s2_addr[:6]}"
        )


if __name__ == "__main__":
    main()