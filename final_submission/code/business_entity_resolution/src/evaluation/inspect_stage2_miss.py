import sys
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline
from src.candidate_filter.cheap_filter import (
    build_candidate_filter_indexes,
    keep_candidate,
    name_tokens,
    address_tokens,
    normalize_simple,
)


DATA_ROOT = Path(
    r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
)

TRAIN_DIR = DATA_ROOT / "dataset" / "train"

TARGET_S1 = "S1-775361341"
TARGET_S3 = "S3-445453666"


def numeric_parts(value):
    value = normalize_simple(value)

    if not value:
        return set()

    return set(
        re.findall(r"\d+", value)
    )


def main():

    print("=" * 70)
    print("INSPECT STAGE-2 FALSE NEGATIVE")
    print("=" * 70)

    print("\nLoading S1...")

    s1 = pipeline.load_source(
        str(TRAIN_DIR / "train_source1.tsv")
    )

    print("Loading S3...")

    s3 = pipeline.load_source(
        str(TRAIN_DIR / "train_source3.tsv")
    )

    # ---------------------------------------------------------
    # Find target records
    # ---------------------------------------------------------

    s1_match = s1[
        s1["entity_id"] == TARGET_S1
    ]

    s3_match = s3[
        s3["entity_id"] == TARGET_S3
    ]

    if s1_match.empty:
        print(f"ERROR: {TARGET_S1} not found.")
        return

    if s3_match.empty:
        print(f"ERROR: {TARGET_S3} not found.")
        return

    s1_row = s1_match.iloc[0]
    s3_row = s3_match.iloc[0]

    # ---------------------------------------------------------
    # Raw records
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCE 1 RECORD")
    print("=" * 70)

    print(f"Entity ID: {s1_row['entity_id']}")
    print(f"Business name: {s1_row['business_name']}")
    print(f"Business address: {s1_row['business_address']}")
    print(f"Country: {s1_row['country']}")

    print("\n" + "=" * 70)
    print("SOURCE 3 RECORD")
    print("=" * 70)

    print(f"Entity ID: {s3_row['entity_id']}")
    print(f"Business name: {s3_row['business_name']}")
    print(f"Business address: {s3_row['business_address']}")
    print(f"Country: {s3_row['country']}")

    # ---------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------

    name1 = normalize_simple(
        s1_row["business_name"]
    )

    name2 = normalize_simple(
        s3_row["business_name"]
    )

    address1 = normalize_simple(
        s1_row["business_address"]
    )

    address2 = normalize_simple(
        s3_row["business_address"]
    )

    # ---------------------------------------------------------
    # Tokens
    # ---------------------------------------------------------

    n1 = name_tokens(name1)
    n2 = name_tokens(name2)

    a1 = address_tokens(address1)
    a2 = address_tokens(address2)

    number1 = numeric_parts(address1)
    number2 = numeric_parts(address2)

    shared_address = a1 & a2
    shared_numbers = number1 & number2

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ADDRESS ANALYSIS")
    print("=" * 70)

    print(
        f"\nNormalized address 1:\n{address1}"
    )

    print(
        f"\nNormalized address 2:\n{address2}"
    )

    print(
        f"\nAddress tokens 1:\n{sorted(a1)}"
    )

    print(
        f"\nAddress tokens 2:\n{sorted(a2)}"
    )

    print(
        f"\nShared address tokens:\n"
        f"{sorted(shared_address)}"
    )

    print(
        f"\nNumeric components 1:\n"
        f"{sorted(number1)}"
    )

    print(
        f"\nNumeric components 2:\n"
        f"{sorted(number2)}"
    )

    print(
        f"\nShared numeric components:\n"
        f"{sorted(shared_numbers)}"
    )

    # ---------------------------------------------------------
    # Name script analysis
    # ---------------------------------------------------------

    latin_chars_1 = len(
        re.findall(r"[a-z]", name1)
    )

    latin_chars_2 = len(
        re.findall(r"[a-z]", name2)
    )

    print("\n" + "=" * 70)
    print("NAME ANALYSIS")
    print("=" * 70)

    print(
        f"\nName tokens 1:\n{sorted(n1)}"
    )

    print(
        f"\nName tokens 2:\n{sorted(n2)}"
    )

    print(
        f"\nShared name tokens:\n"
        f"{sorted(n1 & n2)}"
    )

    print(
        f"\nLatin characters in name 1: "
        f"{latin_chars_1}"
    )

    print(
        f"Latin characters in name 2: "
        f"{latin_chars_2}"
    )

    # ---------------------------------------------------------
    # Country
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("COUNTRY")
    print("=" * 70)

    print(
        f"\nS1 country: {s1_row['country']}"
    )

    print(
        f"S3 country: {s3_row['country']}"
    )

    print(
        f"Country equal: "
        f"{s1_row['country_norm'] == s3_row['country_norm']}"
    )

    # ---------------------------------------------------------
    # Hypothetical rule
    # ---------------------------------------------------------

    strong_numeric_address = (
        len(shared_numbers) >= 1
        and len(shared_address) >= 1
    )

    print("\n" + "=" * 70)
    print("TARGETED RULE TEST")
    print("=" * 70)

    print(
        "\nRule:"
        "\n  At least one shared numeric component"
        "\n  AND at least one shared address token"
    )

    print(
        f"\nWould recover this pair: "
        f"{strong_numeric_address}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()