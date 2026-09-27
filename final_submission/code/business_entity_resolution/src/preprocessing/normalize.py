import re
import unicodedata


# ---------------------------------------------------------
# Generic text normalization
# ---------------------------------------------------------

def normalize_text(value):
    """
    General normalization for business names and addresses.

    Steps:
    1. Handle missing values
    2. Unicode normalization
    3. Convert to lowercase
    4. Normalize common punctuation
    5. Collapse whitespace
    """

    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Replace common punctuation/separators with spaces
    value = re.sub(r"[\.,;:/\\|_\-]+", " ", value)

    # Remove brackets while keeping their contents
    value = re.sub(r"[\(\)\[\]\{\}]", " ", value)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ---------------------------------------------------------
# Business name normalization
# ---------------------------------------------------------

def normalize_business_name(value):
    """
    Normalize a business name.

    This intentionally does NOT remove legal suffixes yet.
    We will test their effect before deciding whether
    to remove them during blocking.
    """

    return normalize_text(value)


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_address(value):
    """
    Normalize a business address.
    """

    value = normalize_text(value)

    if not value:
        return ""

    # Treat literal 'null' as missing
    if value in {"null", "none", "nan"}:
        return ""

    return value


# ---------------------------------------------------------
# Country normalization
# ---------------------------------------------------------

def normalize_country(value):
    """
    Normalize country labels without assuming a fixed
    set of countries.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    if value in {"", "null", "none", "nan"}:
        return ""

    return value


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize(value):
    """
    Convert normalized text into tokens.
    """

    value = normalize_text(value)

    if not value:
        return []

    return value.split()


# ---------------------------------------------------------
# Character n-gram representation
# ---------------------------------------------------------

def character_ngrams(value, n=3):
    """
    Generate character n-grams.

    Useful for noisy strings where words may contain
    spelling errors or small variations.
    """

    value = normalize_text(value)

    if len(value) < n:
        return [value] if value else []

    return [
        value[i:i + n]
        for i in range(len(value) - n + 1)
    ]