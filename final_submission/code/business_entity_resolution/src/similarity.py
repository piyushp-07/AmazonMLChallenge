from typing import Tuple
from collections import Counter
import pandas as pd
from difflib import SequenceMatcher
from math import sqrt
try:
    from rapidfuzz.distance import Levenshtein as _FastLevenshtein
except ImportError:  # Exact pure-Python fallback for environments without RapidFuzz.
    _FastLevenshtein = None


def jaro_winkler(a: str, b: str) -> float:
    # fallback to SequenceMatcher ratio as a cheap proxy
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _FastLevenshtein is not None:
        distance = _FastLevenshtein.distance(a, b)
        return 1.0 - distance / max(len(a), len(b))
    # simple Levenshtein via dynamic programming
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    dist = dp[lb]
    maxlen = max(la, lb)
    return 1.0 - dist / maxlen


def token_jaccard(a: str, b: str) -> float:
    sa = set(a.split()) if a else set()
    sb = set(b.split()) if b else set()
    if not sa and not sb:
        return 1.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0


def token_overlap_count(a: str, b: str) -> int:
    sa = set(a.split()) if a else set()
    sb = set(b.split()) if b else set()
    return len(sa & sb)


def tfidf_cosine(a: str, b: str) -> float:
    # The original two-row vector calculation is equivalent to count-vector
    # cosine. Avoid dense NumPy allocations for every candidate pair.
    if not a and not b:
        return 1.0
    va = Counter(a.split())
    vb = Counter(b.split())
    dot = sum(count * vb.get(token, 0) for token, count in va.items())
    norm_a = sqrt(sum(count * count for count in va.values()))
    norm_b = sqrt(sum(count * count for count in vb.values()))
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return float(dot / denom)
