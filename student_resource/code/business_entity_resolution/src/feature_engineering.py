from typing import Dict
import pandas as pd
import numpy as np
from .normalization import normalize_name, normalize_address
from .similarity import (
    jaro_winkler,
    normalized_levenshtein,
    token_jaccard,
    token_overlap_count,
    tfidf_cosine,
)


def build_features(s1_row: pd.Series, cand_row: pd.Series) -> Dict:
    n1 = normalize_name(s1_row.get("business_name", ""))
    n2 = normalize_name(cand_row.get("business_name", ""))
    a1 = normalize_address(s1_row.get("business_address", ""))
    a2 = normalize_address(cand_row.get("business_address", ""))
    features = {}
    # name features
    features["name_exact"] = 1.0 if n1 and n2 and n1 == n2 else 0.0
    features["name_jw"] = jaro_winkler(n1, n2)
    features["name_lev"] = normalized_levenshtein(n1, n2)
    features["name_jaccard"] = token_jaccard(n1, n2)
    features["name_tok_overlap"] = token_overlap_count(n1, n2)
    features["name_tfidf"] = tfidf_cosine(n1, n2)
    features["name_len_diff"] = abs(len(n1) - len(n2))
    features["name_len_ratio"] = (len(n1) + 1) / (len(n2) + 1) if n2 else 0.0

    # address features
    features["addr_exact"] = 1.0 if a1 and a2 and a1 == a2 else 0.0
    features["addr_jw"] = jaro_winkler(a1, a2)
    features["addr_lev"] = normalized_levenshtein(a1, a2)
    features["addr_jaccard"] = token_jaccard(a1, a2)
    features["addr_tok_overlap"] = token_overlap_count(a1, a2)
    features["addr_tfidf"] = tfidf_cosine(a1, a2)
    features["addr_len_diff"] = abs(len(a1) - len(a2))
    features["addr_len_ratio"] = (len(a1) + 1) / (len(a2) + 1) if a2 else 0.0

    # country
    c1 = (s1_row.get("country") or "").strip().lower()
    c2 = (cand_row.get("country") or "").strip().lower()
    features["country_eq"] = 1.0 if c1 and c2 and c1 == c2 else 0.0
    features["country_missing"] = 1.0 if not c2 else 0.0

    # cross features
    features["name_x_addr"] = features["name_jw"] * features["addr_jw"]
    features["name_plus_addr"] = features["name_jw"] + features["addr_jw"]
    features["both_strong"] = 1.0 if features["name_jw"] > 0.9 and features["addr_jw"] > 0.8 else 0.0

    return features


def features_from_pairs(s1_df: pd.DataFrame, candidates_df: pd.DataFrame, s2s3_df: pd.DataFrame) -> pd.DataFrame:
    # candidates_df expected: columns source1_entity_id, candidate_entity_ids (comma-separated)
    rows = []
    # index s2/s3 by id for fast lookup
    s2s3_index = s2s3_df.set_index("entity_id")
    for _, row in candidates_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_list = [c.strip() for c in (row.get("candidate_entity_ids") or "").split(",") if c.strip()]
        s1_row = s1_df.loc[s1_df["entity_id"] == s1_id].iloc[0]
        for cid in cand_list:
            if cid not in s2s3_index.index:
                continue
            cand_row = s2s3_index.loc[cid]
            feats = build_features(s1_row, cand_row)
            feats.update({"source1_entity_id": s1_id, "candidate_id": cid})
            rows.append(feats)
    return pd.DataFrame(rows)
