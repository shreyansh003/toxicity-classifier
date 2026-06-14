"""
app.py
------
Flask web application for the Text-Based Toxicity Classifier.

Routes
------
GET  /          — Main page (comment input form)
POST /predict   — JSON endpoint returning toxicity prediction
GET  /results   — Model evaluation results page
GET  /health    — Health-check endpoint (JSON)
"""

import os
import sys
import json
import logging

from flask import Flask, render_template, request, jsonify, abort

# Ensure src/ is on the path so predict.py can import preprocess
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import predict_toxicity   # noqa: E402

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper — load saved evaluation metrics
# ---------------------------------------------------------------------------
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "model", "results.json")
EVAL_PATH    = os.path.join(os.path.dirname(__file__), "model", "evaluation.json")


def _load_json_safe(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main comment-analysis page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a JSON body ``{"text": "..."}`` and return prediction JSON.

    Response schema
    ---------------
    {
        "label":       "Toxic" | "Non-Toxic",
        "toxic":       true | false,
        "confidence":  0.97,
        "pct":         "97.00%"
    }
    """
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    raw_text = str(payload["text"]).strip()
    if not raw_text:
        return jsonify({"error": "Text field must not be empty."}), 400

    if len(raw_text) > 5000:
        return jsonify({"error": "Text exceeds maximum length of 5 000 characters."}), 400

    try:
        result = predict_toxicity(raw_text)
    except FileNotFoundError as exc:
        logger.error("Model artefacts missing: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:           # noqa: BLE001
        logger.exception("Prediction error")
        return jsonify({"error": "Internal prediction error."}), 500

    return jsonify({
        "label":      result["label"],
        "toxic":      result["toxic"],
        "confidence": result["confidence"],
        "pct":        f"{result['confidence'] * 100:.2f}%",
    })


@app.route("/results")
def results():
    """Render the model results / analytics page."""
    training_results = _load_json_safe(RESULTS_PATH)
    eval_metrics     = _load_json_safe(EVAL_PATH)
    return render_template(
        "index.html",
        page="results",
        training_results=training_results,
        eval_metrics=eval_metrics,
    )


@app.route("/health")
def health():
    """Simple health-check used by load balancers / CI."""
    return jsonify({"status": "ok", "service": "toxicity-classifier"})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting Toxicity Classifier on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
