import sys
sys.path.insert(0, "./")
from src.data_loader import load_test_sources
from src.feature_engineering import features_from_pairs
import pandas as pd


def build_mock_candidates(s1, s2, s3, n_s1=10, k=3):
    rows = []
    s2_ids = s2["entity_id"].tolist()[:k]
    s3_ids = s3["entity_id"].tolist()[:k]
    for i, eid in enumerate(s1["entity_id"].tolist()[:n_s1]):
        cand = s2_ids + s3_ids
        rows.append({"source1_entity_id": eid, "candidate_entity_ids": ",".join(cand)})
    return pd.DataFrame(rows)


def main():
    s1, s2, s3 = load_test_sources()
    s1_small = s1.head(20)
    s2_small = s2.head(20)
    s3_small = s3.head(20)
    cand = build_mock_candidates(s1_small, s2_small, s3_small, n_s1=10, k=3)
    s2s3 = pd.concat([s2_small, s3_small], ignore_index=True)
    feats = features_from_pairs(s1_small, cand, s2s3)
    print("Features shape:", feats.shape)
    print(feats.head().to_string())


if __name__ == "__main__":
    main()
