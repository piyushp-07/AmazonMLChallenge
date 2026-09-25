import re
import unicodedata
from typing import Optional
import pandas as pd

LEGAL_SUFFIXES = ["ltd", "limited", "pvt", "private", "corp", "corporation", "inc", "llc", "co", "plc"]


def _unicode_norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s) if s else s


def normalize_name(name: Optional[str]) -> str:
    if not name or pd.isna(name):
        return ""
    s = _unicode_norm(name)
    s = s.lower()
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # remove legal suffixes
    parts = s.split()
    if parts and parts[-1] in LEGAL_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def normalize_address(addr: Optional[str]) -> str:
    if not addr or pd.isna(addr):
        return ""
    s = _unicode_norm(addr)
    s = s.lower()
    s = s.replace("street", " st ")
    s = s.replace("road", " rd ")
    s = s.replace("avenue", " ave ")
    s = s.replace("apartment", " apt ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
