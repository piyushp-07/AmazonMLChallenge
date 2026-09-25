from typing import Tuple
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from math import sqrt


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
    # tiny in-memory TF-IDF for two strings
    if not a and not b:
        return 1.0
    toks = [t for t in (a + " " + b).split()]
    vocab = {t: i for i, t in enumerate(sorted(set(toks)))}
    def vec(s):
        v = np.zeros(len(vocab))
        for t in s.split():
            v[vocab[t]] += 1
        return v
    va, vb = vec(a), vec(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
