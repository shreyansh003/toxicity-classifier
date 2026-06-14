"""
predict.py
----------
Inference module for the Toxicity Classifier.

Public API
----------
predict_toxicity(text: str) -> dict
    Returns label ("Toxic" / "Non-Toxic"), confidence, and cleaned text.

predict_batch(texts: list[str]) -> list[dict]
    Vectorised batch prediction.
"""

import os
import logging
import joblib
import numpy as np

from preprocess import clean_text

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

# ---------------------------------------------------------------------------
# Lazy singleton loader
# ---------------------------------------------------------------------------
_model      = None
_vectorizer = None


def _load_artefacts():
    """Load model and vectorizer once and cache in module globals."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run 'python src/train.py' first."
            )
        if not os.path.exists(VEC_PATH):
            raise FileNotFoundError(
                f"Vectorizer not found at {VEC_PATH}. "
                "Run 'python src/train.py' first."
            )
        logger.info("Loading model artefacts …")
        _model      = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VEC_PATH)
        logger.info("Artefacts loaded successfully.")
    return _model, _vectorizer


# ---------------------------------------------------------------------------
# Confidence helper
# ---------------------------------------------------------------------------

def _get_confidence(model, X_vec) -> float:
    """
    Return a confidence score (0–1) for the predicted class.

    * Uses ``predict_proba`` when available (Logistic Regression, NB).
    * Falls back to decision function margin for LinearSVC.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_vec)[0]
        return float(np.max(proba))

    if hasattr(model, "decision_function"):
        margin = model.decision_function(X_vec)[0]
        # Convert raw margin to a pseudo-probability via sigmoid
        confidence = 1 / (1 + np.exp(-np.abs(margin)))
        return float(confidence)

    return 1.0   # last resort


# ---------------------------------------------------------------------------
# Public prediction functions
# ---------------------------------------------------------------------------

def predict_toxicity(text: str) -> dict:
    """
    Predict whether *text* is toxic.

    Parameters
    ----------
    text : str
        Raw user comment.

    Returns
    -------
    dict with keys:
        * ``label``       – "Toxic" or "Non-Toxic"
        * ``toxic``       – bool
        * ``confidence``  – float 0–1
        * ``cleaned_text``– the preprocessed version of the input
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    model, vectorizer = _load_artefacts()

    cleaned   = clean_text(text)
    X_vec     = vectorizer.transform([cleaned])
    prediction = model.predict(X_vec)[0]
    confidence = _get_confidence(model, X_vec)

    is_toxic = bool(prediction == 1)
    label    = "Toxic" if is_toxic else "Non-Toxic"

    return {
        "label":        label,
        "toxic":        is_toxic,
        "confidence":   round(confidence, 4),
        "cleaned_text": cleaned,
    }


def predict_batch(texts: list) -> list:
    """
    Batch predict toxicity for a list of comment strings.

    Parameters
    ----------
    texts : list[str]
        A list of raw comment strings.

    Returns
    -------
    list[dict]
        One result dict per input text (same schema as ``predict_toxicity``).
    """
    if not texts:
        return []

    model, vectorizer = _load_artefacts()

    cleaned_texts = [clean_text(t) for t in texts]
    X_vec         = vectorizer.transform(cleaned_texts)
    predictions   = model.predict(X_vec)

    results = []
    for text, cleaned, pred in zip(texts, cleaned_texts, predictions):
        # Confidence per sample
        X_single = vectorizer.transform([cleaned])
        conf     = _get_confidence(model, X_single)

        is_toxic = bool(pred == 1)
        results.append({
            "label":        "Toxic" if is_toxic else "Non-Toxic",
            "toxic":        is_toxic,
            "confidence":   round(conf, 4),
            "cleaned_text": cleaned,
        })

    return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    samples = [
        "You are an idiot and I hate you!",
        "Thank you for your helpful contribution to this discussion.",
        "This is the worst garbage I have ever read.",
        "Great article, very informative and well written.",
    ]

    if len(sys.argv) > 1:
        samples = [" ".join(sys.argv[1:])]

    print("\n" + "=" * 60)
    print("  TOXICITY CLASSIFIER — DEMO PREDICTIONS")
    print("=" * 60)
    for sample in samples:
        result = predict_toxicity(sample)
        bar = "🔴" if result["toxic"] else "🟢"
        print(f"\n  {bar}  [{result['label']}]  (conf: {result['confidence']:.2%})")
        print(f"     Input   : {sample[:80]}")
        print(f"     Cleaned : {result['cleaned_text'][:80]}")
