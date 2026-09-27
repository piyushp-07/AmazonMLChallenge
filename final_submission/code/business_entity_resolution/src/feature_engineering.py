from typing import Dict
from functools import lru_cache
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


@lru_cache(maxsize=250_000)
def _pair_features(left: str, right: str):
    """Cache deterministic string-pair features without changing their values."""
    exact = 1.0 if left and right and left == right else 0.0
    jw = jaro_winkler(left, right)
    lev = normalized_levenshtein(left, right)
    jaccard = token_jaccard(left, right)
    overlap = token_overlap_count(left, right)
    tfidf = tfidf_cosine(left, right)
    length_difference = abs(len(left) - len(right))
    length_ratio = (len(left) + 1) / (len(right) + 1) if right else 0.0
    return exact, jw, lev, jaccard, overlap, tfidf, length_difference, length_ratio


def build_features(s1_row: pd.Series, cand_row: pd.Series) -> Dict:
    n1 = normalize_name(s1_row.get("business_name", ""))
    n2 = normalize_name(cand_row.get("business_name", ""))
    a1 = normalize_address(s1_row.get("business_address", ""))
    a2 = normalize_address(cand_row.get("business_address", ""))
    features = {}
    # Name and address feature values match the original implementation;
    # caching avoids recomputing them for repeated business strings.
    name_values = _pair_features(n1, n2)
    for key, value in zip(("name_exact", "name_jw", "name_lev", "name_jaccard", "name_tok_overlap", "name_tfidf", "name_len_diff", "name_len_ratio"), name_values):
        features[key] = value

    # address features
    address_values = _pair_features(a1, a2)
    for key, value in zip(("addr_exact", "addr_jw", "addr_lev", "addr_jaccard", "addr_tok_overlap", "addr_tfidf", "addr_len_diff", "addr_len_ratio"), address_values):
        features[key] = value

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


def features_from_pairs(
    s1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    s1_lookup: Dict | None = None,
    candidate_lookup: Dict | None = None,
) -> pd.DataFrame:
    rows = []
    s1_dict = s1_lookup or s1_df.set_index("entity_id").to_dict(orient="index")
    s2s3_dict = candidate_lookup or s2s3_df.set_index("entity_id").to_dict(orient="index")
    empty_record = {"business_name": "", "business_address": "", "country": ""}

    if "candidate_entity_ids" in candidates_df.columns:
        for _, row in candidates_df.iterrows():
            s1_id = str(row["source1_entity_id"])
            cand_list = [c.strip() for c in (row.get("candidate_entity_ids") or "").split(",") if c.strip()]
            s1_row = s1_dict.get(s1_id, empty_record)
            for cid in cand_list:
                if cid not in s2s3_dict:
                    continue
                cand_row = s2s3_dict[cid]
                feats = build_features(s1_row, cand_row)
                feats.update({"source1_entity_id": s1_id, "candidate_id": cid})
                rows.append(feats)
    else:
        cand_col = "candidate_id" if "candidate_id" in candidates_df.columns else "matched_entity_id"
        s1_ids = candidates_df["source1_entity_id"].astype(str).tolist()
        cand_ids = candidates_df[cand_col].astype(str).tolist()
        for s1_id, cid in zip(s1_ids, cand_ids):
            if s1_id not in s1_dict or cid not in s2s3_dict:
                continue
            s1_row = s1_dict[s1_id]
            cand_row = s2s3_dict[cid]
            feats = build_features(s1_row, cand_row)
            feats.update({"source1_entity_id": s1_id, "candidate_id": cid})
            rows.append(feats)

    return pd.DataFrame(rows)


def build_feature_matrix(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate feature matrix for candidate pairs.
    Each row of candidates_df produces one row of features.
    """
    s1_dict = s1_df.set_index("entity_id").to_dict(orient="index")
    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
    s2s3_dict = s2s3_df.set_index("entity_id").to_dict(orient="index")

    empty_record = {"business_name": "", "business_address": "", "country": ""}

    cand_col = "matched_entity_id" if "matched_entity_id" in candidates_df.columns else "candidate_id"

    s1_ids = candidates_df["source1_entity_id"].astype(str).tolist()
    cand_ids = candidates_df[cand_col].astype(str).tolist()

    feature_rows = []
    for s1_id, c_id in zip(s1_ids, cand_ids):
        s1_row = s1_dict.get(s1_id, empty_record)
        cand_row = s2s3_dict.get(c_id, empty_record)
        feature_rows.append(build_features(s1_row, cand_row))

    return pd.DataFrame(feature_rows)

