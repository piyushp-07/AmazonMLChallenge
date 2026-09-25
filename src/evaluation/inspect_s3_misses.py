import pandas as pd

from src.blocking.candidate_generation import load_source


BASE = r"C:\Users\shiva\Downloads\6ab10eb3b23ba_student_resource\student_resource"
TRAIN = BASE + r"\dataset\train"

S1_PATH = TRAIN + r"\train_source1.tsv"
S3_PATH = TRAIN + r"\train_source3.tsv"
GT_PATH = TRAIN + r"\train_ground_truth.tsv"


MISSED_PAIRS = {
    ("S1-261779770", "S3-547868984"),
    ("S1-308240187", "S3-377116427"),
    ("S1-39641619", "S3-245091850"),
    ("S1-800482427", "S3-355590735"),
}


def main():
    print("=" * 70)
    print("INSPECT S1 -> S3 MISSED PAIRS")
    print("=" * 70)

    print("\nLoading Source 1...")
    s1 = load_source(S1_PATH)

    print("Loading Source 3...")
    s3 = load_source(S3_PATH)

    s1_records = s1[
        s1["entity_id"].isin({x[0] for x in MISSED_PAIRS})
    ].copy()

    s3_records = s3[
        s3["entity_id"].isin({x[1] for x in MISSED_PAIRS})
    ].copy()

    s1_records = s1_records.set_index("entity_id")
    s3_records = s3_records.set_index("entity_id")

    for s1_id, s3_id in sorted(MISSED_PAIRS):
        print("\n" + "-" * 70)
        print(f"S1: {s1_id}")
        print(f"S3: {s3_id}")
        print("-" * 70)

        if s1_id in s1_records.index:
            row = s1_records.loc[s1_id]

            print("\nSOURCE 1")
            print(f"Name:    {row['business_name']}")
            print(f"Address: {row['business_address']}")
            print(f"Country: {row['country']}")

            print("\nNormalized")
            print(f"Name:    {row['name_norm']}")
            print(f"Address: {row['address_norm']}")
            print(f"Country: {row['country_norm']}")

        if s3_id in s3_records.index:
            row = s3_records.loc[s3_id]

            print("\nSOURCE 3")
            print(f"Name:    {row['business_name']}")
            print(f"Address: {row['business_address']}")
            print(f"Country: {row['country']}")

            print("\nNormalized")
            print(f"Name:    {row['name_norm']}")
            print(f"Address: {row['address_norm']}")
            print(f"Country: {row['country_norm']}")


if __name__ == "__main__":
    main()