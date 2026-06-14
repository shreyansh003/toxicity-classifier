"""
preprocess.py
-------------
Text preprocessing pipeline for the Toxicity Classifier.
Handles cleaning, normalization, and preparation of raw comment text.
"""

import re
import string
import logging
import pandas as pd
import numpy as np

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning helpers
# ---------------------------------------------------------------------------

def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www.* links."""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub("", text)


def remove_punctuation(text: str) -> str:
    """Strip all punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline applied to a single string:
      1. Lowercase
      2. Remove URLs
      3. Remove punctuation
      4. Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV dataset from *filepath*.

    Returns a DataFrame with only the columns we need:
    ``comment_text`` and ``toxic``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the required columns are not present.
    """
    logger.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath)

    required_cols = {"comment_text", "toxic"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    logger.info("Loaded %d rows.", len(df))
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare a raw DataFrame for modelling.

    Steps
    -----
    * Drop rows where ``comment_text`` or ``toxic`` is null.
    * Apply ``clean_text`` to ``comment_text``.
    * Drop any rows where the cleaned text is empty.
    * Cast ``toxic`` to int (0 / 1).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from ``load_data``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with a new ``cleaned_text`` column.
    """
    initial_rows = len(df)
    logger.info("Starting preprocessing — %d rows.", initial_rows)

    # 1. Drop nulls in relevant columns
    df = df.dropna(subset=["comment_text", "toxic"]).copy()
    logger.info("After dropping nulls: %d rows.", len(df))

    # 2. Clean text
    df["cleaned_text"] = df["comment_text"].apply(clean_text)

    # 3. Remove rows where cleaned text is empty
    df = df[df["cleaned_text"].str.strip() != ""]
    logger.info("After removing empty text: %d rows.", len(df))

    # 4. Ensure target is integer
    df["toxic"] = df["toxic"].astype(int)

    removed = initial_rows - len(df)
    logger.info("Preprocessing complete. Removed %d rows total.", removed)
    return df


# ---------------------------------------------------------------------------
# Quick sanity check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    df = load_data(path)
    df_clean = preprocess_dataframe(df)
    print(df_clean[["comment_text", "cleaned_text", "toxic"]].head(5))
    print(f"\nClass distribution:\n{df_clean['toxic'].value_counts()}")
