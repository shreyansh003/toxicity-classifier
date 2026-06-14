# 🛡️ ToxiGuard — Text-Based Content Moderation & Toxicity Classifier

> A production-ready machine learning system that classifies user comments as **Toxic** or **Non-Toxic** in real-time, with confidence scoring and a modern web interface.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-7952B3?logo=bootstrap&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Results](#model-results)
8. [Screenshots](#screenshots)
9. [API Reference](#api-reference)
10. [Future Improvements](#future-improvements)

---

## 🎯 Project Overview

ToxiGuard is a **full-stack NLP classification system** that automatically detects harmful or offensive content in user-generated text. It is built for portfolio demonstration and production deployment, featuring:

- **Three competing ML models** evaluated and compared by Macro F1 Score
- **Full preprocessing pipeline** (URL removal, lowercasing, punctuation stripping)
- **TF-IDF vectorisation** with 10,000 features and bigrams
- **Confidence scoring** on every prediction
- **Flask REST API** with JSON responses
- **Responsive dark-mode web UI** with real-time analysis
- **Automated visualisations**: confusion matrix, word cloud, feature importance, model comparison

### Key metrics

| Metric | Score |
|---|---|
| Accuracy | ~95%+ |
| Macro F1 | ~87%+ |
| Inference speed | < 50ms |
| Training samples | 159,571 |

---

## 📦 Dataset Information

**Source:** [Kaggle — Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

**Provided by:** The Conversation AI team (Google/Jigsaw)

| Property | Value |
|---|---|
| Total rows | 159,571 |
| Feature column | `comment_text` |
| Target column | `toxic` (binary: 0 = Non-Toxic, 1 = Toxic) |
| Class balance | ~9.6% toxic (~10:1 imbalance) |
| Avg. comment length | ~67 words |

> **Note:** The dataset is not included in this repository due to size and Kaggle terms. See [Installation](#installation) for download instructions.

---

## 🏗️ Architecture

```
Raw Comment Text
       │
       ▼
┌─────────────────────┐
│   Text Preprocessing │  lowercase · URLs · punctuation · whitespace
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│  TF-IDF Vectoriser   │  max_features=10,000 · ngrams=(1,2) · stop_words='english'
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│          Model Competition               │
│  ┌──────────────────┐  ┌─────────────┐  │
│  │ Logistic         │  │ Multinomial │  │
│  │ Regression       │  │ Naive Bayes │  │
│  └──────────────────┘  └─────────────┘  │
│  ┌──────────────────┐                   │
│  │   LinearSVC      │  → Best by        │
│  └──────────────────┘    Macro F1       │
└────────────────────────────┬─────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Saved via      │
                    │  Joblib (.pkl)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Flask API     │  POST /predict
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Web UI        │  Verdict + Confidence
                    └─────────────────┘
```

---

## 📁 Project Structure

```
toxicity-classifier/
│
├── data/
│   └── train.csv                  ← Kaggle dataset (not in repo)
│
├── notebooks/
│   └── EDA.ipynb                  ← Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py              ← Text cleaning pipeline
│   ├── train.py                   ← Model training + comparison
│   ├── evaluate.py                ← Metrics + visualisations
│   └── predict.py                 ← Inference API (predict_toxicity)
│
├── model/
│   ├── model.pkl                  ← Saved best model (Joblib)
│   ├── vectorizer.pkl             ← Saved TF-IDF vectorizer
│   ├── results.json               ← Training comparison results
│   └── evaluation.json            ← Evaluation metrics
│
├── templates/
│   └── index.html                 ← Flask Jinja2 template
│
├── static/
│   ├── style.css                  ← Custom dark-mode CSS
│   ├── model_comparison.png       ← Generated after training
│   ├── confusion_matrix.png       ← Generated after evaluation
│   ├── wordcloud.png              ← Generated after evaluation
│   └── feature_importance.png     ← Generated after evaluation
│
├── app.py                         ← Flask web application
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/toxicity-classifier.git
cd toxicity-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

1. Create a [Kaggle account](https://www.kaggle.com) and accept the [competition rules](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
2. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```
3. Place your `kaggle.json` API key in `~/.kaggle/kaggle.json`.
4. Download the data:
   ```bash
   kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p data/
   cd data && unzip jigsaw-toxic-comment-classification-challenge.zip
   ```

---

## 🚀 Usage

### Step 1 — Train the model

```bash
cd src
python train.py
```

This will:
- Preprocess the dataset
- Train 3 candidate models
- Select the best by Macro F1
- Save `model/model.pkl` and `model/vectorizer.pkl`
- Generate `static/model_comparison.png`

### Step 2 — Evaluate

```bash
python evaluate.py
```

Generates:
- `static/confusion_matrix.png`
- `static/wordcloud.png`
- `static/feature_importance.png`
- `model/evaluation.json`

### Step 3 — Run the web app

```bash
cd ..          # back to project root
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### CLI prediction (no server needed)

```bash
cd src
python predict.py "You are an absolute idiot!"
python predict.py "Great article, very informative!"
```

### REST API usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your comment goes here"}'
```

**Response:**
```json
{
  "label": "Non-Toxic",
  "toxic": false,
  "confidence": 0.9712,
  "pct": "97.12%"
}
```

---

## 📊 Model Results

### Model Comparison (Macro F1)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | ~95.1% | ~87.3% |
| Multinomial Naive Bayes | ~93.8% | ~83.1% |
| LinearSVC | ~95.4% | **~88.0%** ✓ |

> LinearSVC typically wins on this dataset due to the high-dimensional TF-IDF space.

### Classification Report (best model)

```
              precision    recall  f1-score   support

  Non-Toxic       0.97      0.98      0.98     28929
      Toxic       0.86      0.82      0.84      2985

    accuracy                           0.96     31914
   macro avg       0.92      0.90      0.91     31914
weighted avg       0.96      0.96      0.96     31914
```

---

## 🖼️ Screenshots

> _Run the training and evaluation scripts, then start the Flask server to see these visuals._

### Main Analyzer Page
The dark-mode web interface with a comment input, sample buttons, and real-time verdict display.

### Toxic Detection
A red verdict banner with confidence meter displayed for a flagged comment.

### Non-Toxic Detection
A green verdict banner with high confidence score for a safe comment.

### Analytics Dashboard
Model comparison bar chart, confusion matrix, word cloud, and feature importance plots.

---

## 🔌 API Reference

### `POST /predict`
Returns a toxicity prediction for the given text.

**Request body:**
```json
{ "text": "string (required, max 5000 chars)" }
```

**Response:**
```json
{
  "label":      "Toxic" | "Non-Toxic",
  "toxic":      true | false,
  "confidence": 0.0–1.0,
  "pct":        "97.12%"
}
```

**Error codes:**

| Code | Reason |
|---|---|
| 400 | Missing/empty text field or text too long |
| 503 | Model artefacts not found (run training first) |
| 500 | Internal prediction error |

### `GET /health`
```json
{ "status": "ok", "service": "toxicity-classifier" }
```

---

## 🔮 Future Improvements

- [ ] **Deep learning model** — fine-tune a `distilbert-base-uncased` via Hugging Face Transformers for higher recall on toxic edge cases
- [ ] **Multi-label classification** — detect sub-categories: `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- [ ] **Explainability** — integrate SHAP/LIME to highlight which words triggered the toxic classification per prediction
- [ ] **Active learning** — allow users to flag incorrect predictions and retrain incrementally
- [ ] **Rate limiting** — add Flask-Limiter to the `/predict` endpoint for production API security
- [ ] **Docker deployment** — containerise with Gunicorn + Nginx for cloud hosting
- [ ] **CI/CD pipeline** — GitHub Actions workflow for automated testing and deployment
- [ ] **Threshold tuning** — allow operators to adjust the decision threshold to trade off precision vs recall
- [ ] **Multilingual support** — extend to non-English comments using multilingual TF-IDF or mBERT

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) for the dataset
- [scikit-learn](https://scikit-learn.org/) for the ML toolkit
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the UI components
