from pathlib import Path
import pandas as pd


# Change this only if your dataset is stored somewhere else
DATASET_ROOT = Path(
    r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
)

TRAIN_DIR = DATASET_ROOT / "dataset" / "train"
TEST_DIR = DATASET_ROOT / "dataset" / "test"


def inspect_file(path, nrows=5):
    print("\n" + "=" * 80)
    print(f"FILE: {path.name}")
    print("=" * 80)

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=nrows,
        dtype=str
    )

    print("\nColumns:")
    print(list(df.columns))

    print("\nShape of sample:")
    print(df.shape)

    print("\nSample rows:")
    print(df.to_string(index=False))

    print("\nMissing values in sample:")
    print(df.isna().sum())


def inspect_countries(path, chunksize=100_000):
    print("\n" + "=" * 80)
    print(f"COUNTRY DISTRIBUTION: {path.name}")
    print("=" * 80)

    counts = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=["country"],
        chunksize=chunksize
    ):
        values = chunk["country"].fillna("<MISSING>")

        for value, count in values.value_counts().items():
            counts[value] = counts.get(value, 0) + int(count)

    result = pd.Series(counts).sort_values(ascending=False)

    print(result.to_string())
    print(f"\nTotal rows: {result.sum()}")


def inspect_text_lengths(path, chunksize=100_000):
    print("\n" + "=" * 80)
    print(f"TEXT LENGTHS: {path.name}")
    print("=" * 80)

    name_lengths = []
    address_lengths = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=["business_name", "business_address"],
        chunksize=chunksize
    ):
        names = chunk["business_name"].fillna("")
        addresses = chunk["business_address"].fillna("")

        name_lengths.extend(names.str.len().tolist())
        address_lengths.extend(addresses.str.len().tolist())

    name_series = pd.Series(name_lengths)
    address_series = pd.Series(address_lengths)

    print("\nBusiness name:")
    print(name_series.describe())

    print("\nBusiness address:")
    print(address_series.describe())


def main():
    train_files = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
        TRAIN_DIR / "train_ground_truth.tsv",
    ]

    test_files = [
        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]

    print("BUSINESS ENTITY RESOLUTION - DATASET PROFILE")
    print("=" * 80)
    print(f"Dataset root: {DATASET_ROOT}")

    print("\n\nTRAINING FILES")

    for path in train_files:
        inspect_file(path)

    print("\n\nTEST FILES")

    for path in test_files:
        inspect_file(path)

    print("\n\nCOUNTRY ANALYSIS")

    for path in [
        train_files[0],
        train_files[1],
        train_files[2],
        test_files[0],
        test_files[1],
        test_files[2],
    ]:
        inspect_countries(path)

    print("\n\nTEXT LENGTH ANALYSIS")

    for path in [
        train_files[0],
        train_files[1],
        train_files[2],
    ]:
        inspect_text_lengths(path)


if __name__ == "__main__":
    main()