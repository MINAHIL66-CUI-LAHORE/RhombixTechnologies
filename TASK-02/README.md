# Twitter Sentiment Analysis
### Rhombix Technologies — Task II
**Submitted by:** Minahil Aftab

**Domain:** MACHINE LEARNING

---

## Overview

A full end-to-end pipeline for binary sentiment classification (Positive / Negative) of tweets using the **Sentiment140** dataset of 1.6 million tweets. The project covers exploratory data analysis, traditional machine learning with TF-IDF features, and deep learning with a fine-tuned **DistilBERT** transformer — achieving up to **92–93% accuracy**.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Part 1: Setup & Data Loading](#part-1-setup--data-loading)
   - [Part 2: Exploratory Data Analysis](#part-2-exploratory-data-analysis)
   - [Part 3: Text Preprocessing](#part-3-text-preprocessing)
   - [Part 4: Classical ML Models](#part-4-classical-ml-models)
   - [Part 5: DistilBERT Fine-Tuning](#part-5-distilbert-fine-tuning)
   - [Part 6: Topic-Based Sentiment Analysis](#part-6-topic-based-sentiment-analysis)
   - [Part 7: Inference Demo](#part-7-inference-demo)
5. [Results](#results)
6. [Key Findings](#key-findings)
7. [Output Files](#output-files)
8. [Quick Start](#quick-start)

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | Sentiment140 |
| **Size** | 1,600,000 tweets |
| **Labels** | Binary — Positive (1) / Negative (0) |
| **Balance** | Perfectly balanced — 800K each class |
| **Source** | Auto-labelled using emoticons (Go et al., 2009) |
| **Columns** | `polarity`, `id`, `date`, `query`, `user`, `text` |

The dataset is downloaded directly from Google Drive into memory — no manual download needed.

> **Note:** Because labels were auto-assigned from emoticons rather than human annotation, there is an inherent noise ceiling of ~82–84% for traditional ML models on this benchmark.

---

## Project Structure

```
Twitter_Sentiment_Analysis_RHOMBIX_TECHNOLOGIES_TASK_II.ipynb
│
├── Part 1 — Setup & Data Loading
├── Part 2 — Exploratory Data Analysis
├── Part 3 — Text Preprocessing
├── Part 4 — Classical ML Models (~82% accuracy)
├── Part 5 — DistilBERT Fine-Tuning (~92–93% accuracy)
├── Part 6 — Topic-Based Sentiment Analysis
└── Part 7 — Inference Demo
```

---

## Requirements

### Core Libraries (auto-installed in notebook)

```
pandas
numpy
matplotlib
seaborn
scikit-learn
wordcloud
tqdm
gdown
```

### Deep Learning — Part 5 only (manual install)

```bash
pip install torch transformers datasets accelerate
```

> A **GPU is strongly recommended** for Part 5. Training on 160K samples takes ~1–2 hours on GPU and several hours on CPU. Google Colab (free tier) works well.

---

## Pipeline Walkthrough

### Part 1: Setup & Data Loading

- Installs and imports all required libraries.
- Downloads the Sentiment140 dataset (~240 MB) from Google Drive directly into memory using `gdown`.
- Maps polarity labels: `0 → Negative`, `4 → Positive (mapped to 1)`.
- Prints a dataset summary including shape, missing values, and class distribution.

### Part 2: Exploratory Data Analysis

Five visualisations are produced:

| Plot | Description |
|------|-------------|
| **Class Balance** | Bar + pie chart confirming 50/50 split |
| **Tweet Length Distribution** | Character and word length histograms overall and by sentiment |
| **Top Words by Sentiment** | Top 20 most frequent words for positive and negative tweets (stopwords excluded) |
| **Word Clouds** | Visual word clouds in red (negative) and green (positive) palettes |
| **Tweets Over Time** | Monthly tweet volume broken down by sentiment |

### Part 3: Text Preprocessing

A custom preprocessing function is applied to all 1.6M tweets:

| Step | Description |
|------|-------------|
| URL removal | Strips `http://`, `www.`, etc. |
| Mention removal | Removes `@username` tags |
| Hashtag handling | Keeps the word, removes the `#` symbol |
| Character cleaning | Retains letters only; removes punctuation and digits |
| Lowercasing + normalisation | Uniform lowercase, collapse whitespace |
| **Emoticon signal injection** | Detects `:)`, `:(`, `<3`, `XD`, etc. and appends synthetic tokens (`posemoji`, `negemoji`, `exclaim`) to preserve emotional signal |

The cleaned text is stored in a new `clean` column. The dataset is then split 80/20 into training and test sets using stratified sampling.

### Part 4: Classical ML Models

TF-IDF vectorisation is applied with the following configuration:

- `max_features`: 150,000
- `ngram_range`: (1, 2) — unigrams and bigrams
- `sublinear_tf`: True (log-dampened term frequency)
- `min_df`: 4 / `max_df`: 0.90

Four models are trained and compared:

| Model | Description |
|-------|-------------|
| **Complement Naive Bayes** | Fast baseline; works well on text |
| **SGD (Modified Huber)** | Best classical model; supports `predict_proba` |
| **LinearSVC (Calibrated)** | SVM with probability calibration via sigmoid |
| **Ensemble (SGD + NB)** | Soft-voting with 70% SGD / 30% NB weights |

Evaluation includes accuracy, ROC-AUC, confusion matrix, ROC curve, and feature importance (top positive and negative TF-IDF coefficients from the SGD model).

### Part 5: DistilBERT Fine-Tuning

Fine-tunes `distilbert-base-uncased` for sequence classification:

| Setting | Value |
|---------|-------|
| Max token length | 64 |
| Batch size | 64 |
| Epochs | 3 |
| Learning rate | 2e-5 with linear warm-up |
| Training samples | 160,000 (scalable up to 1,280,000) |
| Test samples | 40,000 |

Key implementation notes:
- Uses **raw tweet text** (not preprocessed) — BERT's tokeniser handles it better.
- Gradient clipping at 1.0 for training stability.
- Linear learning rate schedule with 6% warm-up steps.
- Model is saved to `./bert_sentiment_model` after training.
- A training curve (loss + validation accuracy per epoch) is plotted.
- Gracefully skips if `torch`/`transformers` are not installed.

### Part 6: Topic-Based Sentiment Analysis

Uses the best classical model to analyse sentiment around 8 predefined topics by regex-matching tweet text:

`iPhone`, `Twitter`, `Google`, `Monday`, `Weekend`, `Sleep`, `Food`, `Work`

For each topic, up to 5,000 matching tweets are sampled and classified. Results are visualised as grouped bar charts and a net sentiment score chart (positive% − 50%).

### Part 7: Inference Demo

A reusable `SentimentPredictor` class wraps the TF-IDF + classifier pipeline:

```python
predictor = SentimentPredictor(best_clf, tfidf, preprocess)
predictor.print_predictions(["Just had the best coffee ever! :)", "My laptop crashed :("])
```

Output includes sentiment label, confidence score, and emoji indicator. Additional analysis shows the **confidence distribution** of correct vs. incorrect predictions and an **accuracy-vs-threshold** curve showing the trade-off between prediction confidence and coverage.

---

## Results

| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| Complement Naive Bayes | ~80% | — | Fast baseline |
| SGD (Modified Huber) | ~82% | ~0.90 | Best classical model |
| LinearSVC (Calibrated) | ~82% | ~0.90 | SVM-based |
| Ensemble (SGD + NB) | ~82% | ~0.90 | Soft-voting |
| **DistilBERT** | **~92–93%** | — | **Exceeds 90% target** |

Traditional ML models plateau at ~82–83%, consistent with the known noise ceiling of the Sentiment140 benchmark. DistilBERT surpasses 90% accuracy, matching published literature (Go et al., 2009; Rosenthal et al., 2017).

---

## Key Findings

- The Sentiment140 dataset is **perfectly balanced** — no class weighting or resampling is required.
- **Emoticon-based signal tokens** improve classical model performance by preserving emotional cues that are lost during cleaning.
- **Weekend and food** topics show the highest positive sentiment in the dataset.
- **Monday and work** topics skew consistently negative.
- **DistilBERT** trained on just 10% of the data (160K samples) already outperforms all classical models by ~10 percentage points.
- Higher-confidence predictions are significantly more accurate, enabling a configurable confidence threshold for production use.

---

## Output Files

The notebook saves the following plots to disk:

| File | Contents |
|------|----------|
| `class_balance.png` | Bar + pie chart of label distribution |
| `tweet_lengths.png` | Tweet length histograms |
| `top_words.png` | Top 20 words per sentiment class |
| `wordclouds.png` | Word clouds for positive and negative tweets |
| `tweets_over_time.png` | Monthly volume by sentiment |
| `model_comparison.png` | Accuracy, AUC, and training time across models |
| `confusion_roc.png` | Confusion matrix and ROC curve for best model |
| `feature_importance.png` | Top positive and negative TF-IDF n-gram coefficients |
| `bert_training_curve.png` | DistilBERT loss and validation accuracy per epoch |
| `topic_sentiment.png` | Sentiment breakdown and net score by topic |
| `confidence_analysis.png` | Confidence distribution and accuracy-threshold curve |
| `./bert_sentiment_model/` | Saved DistilBERT model and tokeniser weights |

---

## Quick Start

```python
# Run inference on custom tweets after executing the notebook
predictor.print_predictions([
    "This is amazing!",
    "Terrible day.",
    "I hate my life",
    "I am good",
    "I don't like iPhone"
])
```

Example output:
```
======================================================================
 😊 POSITIVE  (87.3% confidence)
   Tweet: This is amazing!
----------------------------------------------------------------------
 😞 NEGATIVE  (91.5% confidence)
   Tweet: Terrible day.
----------------------------------------------------------------------
```

---

