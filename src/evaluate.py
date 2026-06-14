"""
evaluate.py
-----------
Standalone evaluation script for the saved Toxicity Classifier.

Loads the persisted model + vectorizer, runs inference on a held-out
test set (or the validation split), and produces:
  * Accuracy / Precision / Recall / Macro F1
  * Full classification report
  * Confusion matrix (saved as PNG)
  * Word-cloud for toxic comments (saved as PNG)
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
import matplotlib.ticker as ticker

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

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
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VEC_PATH   = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CM_PLOT    = os.path.join(STATIC_DIR, "confusion_matrix.png")
WC_PLOT    = os.path.join(STATIC_DIR, "wordcloud.png")
FI_PLOT    = os.path.join(STATIC_DIR, "feature_importance.png")
EVAL_JSON  = os.path.join(BASE_DIR, "model", "evaluation.json")

os.makedirs(STATIC_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    """Save a styled confusion matrix PNG."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Non-Toxic", "Toxic"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved → %s", save_path)


# ---------------------------------------------------------------------------
# Word cloud
# ---------------------------------------------------------------------------

def plot_wordcloud(df: pd.DataFrame, save_path: str) -> None:
    """
    Generate a word cloud from toxic comments.
    Falls back to a bar chart of top terms if wordcloud is unavailable.
    """
    toxic_text = " ".join(df[df["toxic"] == 1]["cleaned_text"].values)

    try:
        from wordcloud import WordCloud
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="Reds",
            max_words=200,
            collocations=False,
        ).generate(toxic_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Most Common Words in Toxic Comments",
                     fontsize=14, fontweight="bold", pad=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Word cloud saved → %s", save_path)

    except ImportError:
        logger.warning("wordcloud not installed — generating top-terms bar chart instead.")
        _plot_top_terms_bar(toxic_text, save_path)


def _plot_top_terms_bar(text: str, save_path: str, top_n: int = 25) -> None:
    """Fallback: bar chart of top N terms from *text*."""
    from collections import Counter
    import re
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    stopwords = {
        "the", "and", "for", "that", "this", "you", "are", "was",
        "with", "have", "from", "your", "not", "but", "they",
    }
    counts = Counter(w for w in words if w not in stopwords)
    top = counts.most_common(top_n)
    labels, values = zip(*top)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], values[::-1], color="#E53935")
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {top_n} Terms in Toxic Comments", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Top-terms bar chart saved → %s", save_path)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, vectorizer, save_path: str,
                             top_n: int = 20) -> None:
    """
    Plot top-N TF-IDF features for Toxic class.
    Works for LogisticRegression and LinearSVC (coef_).
    Skips silently for models without coef_.
    """
    if not hasattr(model, "coef_"):
        logger.warning("Model has no coef_ attribute — skipping feature importance plot.")
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_

    # For multi-output SVMs coef_ can be 2-D; grab class-1 row
    if coef.ndim > 1:
        coef = coef[0]

    top_idx  = np.argsort(coef)[-top_n:]
      
    top_features = feature_names[top_idx]
    top_weights  = coef[top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#E53935" if w > 0 else "#1E88E5" for w in top_weights]
    ax.barh(range(top_n), top_weights, color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("Coefficient Weight")
    ax.set_title(f"Top {top_n} Features (Toxic → Positive Weight)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance plot saved → %s", save_path)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(data_path: str = DATA_PATH) -> dict:
    """
    Load saved model + vectorizer, build a validation split identical to
    training (same random_state + stratify), run inference and report metrics.
    """
    # Load artefacts
    logger.info("Loading model from: %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("Loading vectorizer from: %s", VEC_PATH)
    vectorizer = joblib.load(VEC_PATH)

    # Load & preprocess data
    df = load_data(data_path)
    df = preprocess_dataframe(df)

    X = df["cleaned_text"].values
    y = df["toxic"].values

    # Reproduce the same val split used during training
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val_tfidf = vectorizer.transform(X_val)
    y_pred = model.predict(X_val_tfidf)

    # Metrics
    metrics = {
        "accuracy":  round(accuracy_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred, average="macro"), 4),
        "recall":    round(recall_score(y_val, y_pred, average="macro"), 4),
        "macro_f1":  round(f1_score(y_val, y_pred, average="macro"), 4),
    }

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k.capitalize():12s}: {v:.4f}")
    print("\n" + classification_report(y_val, y_pred,
                                       target_names=["Non-Toxic", "Toxic"]))

    # Persist metrics
    with open(EVAL_JSON, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Evaluation metrics saved → %s", EVAL_JSON)

    # Visualisations
    plot_confusion_matrix(y_val, y_pred, CM_PLOT)
    plot_wordcloud(df, WC_PLOT)
    plot_feature_importance(model, vectorizer, FI_PLOT)

    return metrics


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    evaluate(data_path=path)
