"""
train.py
--------
Model training pipeline for the Toxicity Classifier.

Trains three candidate models, compares them by Macro F1 Score,
saves the best model + vectorizer via Joblib, and persists a
comparison table to disk.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score

from preprocess import load_data, preprocess_dataframe

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
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
STATIC_DIR  = os.path.join(BASE_DIR, "static")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.pkl")
VEC_PATH    = os.path.join(MODEL_DIR, "vectorizer.pkl")
RESULTS_PATH = os.path.join(MODEL_DIR, "results.json")
COMPARISON_PLOT = os.path.join(STATIC_DIR, "model_comparison.png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
CANDIDATE_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    ),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "LinearSVC": LinearSVC(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    ),
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_vectorizer() -> TfidfVectorizer:
    """Return a configured TfidfVectorizer."""
    return TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        sublinear_tf=True,       # dampens high-frequency terms
        ngram_range=(1, 2),      # unigrams + bigrams for richer features
        min_df=3,                # ignore very rare terms
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(model, X_val, y_val) -> dict:
    """Return a dict of evaluation metrics for *model* on (X_val, y_val)."""
    y_pred = model.predict(X_val)
    return {
        "accuracy":    round(accuracy_score(y_val, y_pred), 4),
        "macro_f1":    round(f1_score(y_val, y_pred, average="macro"), 4),
        "report":      classification_report(y_val, y_pred,
                                             target_names=["Non-Toxic", "Toxic"]),
    }


# ---------------------------------------------------------------------------
# Comparison visualisation
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict, save_path: str) -> None:
    """
    Bar chart comparing Macro F1 scores of all candidate models.
    Saved as a PNG to *save_path*.
    """
    names  = list(results.keys())
    scores = [results[n]["macro_f1"] for n in names]
    colors = ["#4CAF50" if s == max(scores) else "#90A4AE" for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Model Comparison — Macro F1 Score", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Model comparison plot saved to: %s", save_path)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(data_path: str = DATA_PATH) -> None:
    """End-to-end training pipeline."""

    # 1. Load & preprocess
    logger.info("=== Step 1: Load & preprocess data ===")
    df = load_data(data_path)
    df = preprocess_dataframe(df)

    X = df["cleaned_text"].values
    y = df["toxic"].values
    logger.info("Class distribution — Non-Toxic: %d | Toxic: %d",
                (y == 0).sum(), (y == 1).sum())

    # 2. Train / validation split (stratified)
    logger.info("=== Step 2: Train / validation split ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train size: %d  |  Val size: %d", len(X_train), len(X_val))

    # 3. TF-IDF vectorisation
    logger.info("=== Step 3: TF-IDF vectorisation ===")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    logger.info("Vocabulary size: %d", len(vectorizer.vocabulary_))

    # 4. Train & evaluate all candidates
    logger.info("=== Step 4: Train & evaluate candidate models ===")
    results: dict = {}
    for name, clf in CANDIDATE_MODELS.items():
        logger.info("  Training: %s", name)
        clf.fit(X_train_tfidf, y_train)
        metrics = evaluate_model(clf, X_val_tfidf, y_val)
        results[name] = metrics
        logger.info("    Accuracy: %.4f  |  Macro F1: %.4f",
                    metrics["accuracy"], metrics["macro_f1"])

    # 5. Select best model
    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = CANDIDATE_MODELS[best_name]
    logger.info("=== Best model: %s (Macro F1 = %.4f) ===",
                best_name, results[best_name]["macro_f1"])

    # 6. Save model + vectorizer
    logger.info("=== Step 5: Save artefacts ===")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer,  VEC_PATH)
    logger.info("Model saved  → %s", MODEL_PATH)
    logger.info("Vectorizer saved → %s", VEC_PATH)

    # 7. Persist metrics
    serialisable = {
        name: {k: v for k, v in m.items() if k != "report"}
        for name, m in results.items()
    }
    serialisable["best_model"] = best_name
    with open(RESULTS_PATH, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Results JSON saved → %s", RESULTS_PATH)

    # 8. Print final report for the best model
    print("\n" + "=" * 60)
    print(f"  BEST MODEL: {best_name}")
    print("=" * 60)
    print(results[best_name]["report"])

    # 9. Plot comparison
    plot_model_comparison(results, COMPARISON_PLOT)

    logger.info("Training pipeline complete.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    train(data_path=path)
