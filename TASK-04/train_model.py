"""
Fake News Detection System — training script.

Dataset: clmentbisaillon/fake-and-real-news-dataset
  -> ships as two files: True.csv (real) and Fake.csv (fake)

Pipeline: load data -> combine/label -> strip source leakage patterns
-> clean/tokenize/remove stopwords -> TF-IDF vectorize
-> train Logistic Regression, Random Forest, and Naive Bayes -> evaluate
-> save the best model + vectorizer for the app to use.

Run with:  python3 train_model.py
"""

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
RANDOM_STATE = 42

# Patterns that leak the label in this specific dataset. Almost all "real"
# articles here are Reuters wire copy and start with a dateline like
# "WASHINGTON (Reuters) -". If we don't strip these, the model learns
# "has a wire-service dateline" instead of "is factually reliable", and
# then flags any real article that isn't written in that exact style
# (i.e. most real-world news) as fake.
DATELINE_RE = re.compile(
    r"^\s*[A-Z][A-Za-z.,\s]{0,40}\((Reuters|AP|AFP)\)\s*-\s*", flags=re.MULTILINE
)
REUTERS_MENTION_RE = re.compile(r"\breuters\b", flags=re.IGNORECASE)


def strip_leakage(text: str) -> str:
    """Remove wire-service datelines and outlet name mentions that leak the label."""
    text = DATELINE_RE.sub("", text)
    text = REUTERS_MENTION_RE.sub("", text)
    return text


DATA_DIR = BASE_DIR / "data"


def load_data() -> pd.DataFrame:
    """Load True.csv/Fake.csv from the local data/ folder, combine, label, de-leak."""
    true_df = pd.read_csv(DATA_DIR / "True.csv")
    fake_df = pd.read_csv(DATA_DIR / "Fake.csv")
    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).str.strip()
    df["content"] = df["content"].apply(strip_leakage)
    df["label"] = df["label"].str.upper().str.strip()

    # Shuffle — the raw files are label-sorted (all real, then all fake),
    # which otherwise biases the train/test split before stratify kicks in.
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    df = df[df["content"].str.len() > 0].reset_index(drop=True)
    return df[["content", "label"]]


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    df = load_data()
    print(f"Loaded {len(df)} labeled articles")
    print(df["label"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing (tokenize + remove stopwords)")
    print("=" * 60)
    df["clean"] = df["content"].apply(clean_text)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    print("Example before:", df["content"].iloc[0][:120], "...")
    print("Example after: ", df["clean"].iloc[0][:120], "...")

    print("\n" + "=" * 60)
    print("STEP 3: Train / test split")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    print("\n" + "=" * 60)
    print("STEP 4: TF-IDF vectorization")
    print("=" * 60)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\n" + "=" * 60)
    print("STEP 5: Train + evaluate classification models")
    print("=" * 60)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Naive Bayes": MultinomialNB(),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        trained[name] = model
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))
        print("Confusion matrix (rows=actual, cols=predicted):")
        print(confusion_matrix(y_test, preds, labels=["REAL", "FAKE"]))

    print("\n" + "=" * 60)
    print("STEP 6: Save best model")
    print("=" * 60)
    best_name = max(results, key=results.get)
    best_model = trained[best_name]
    print(f"Best model: {best_name} (accuracy={results[best_name]:.4f})")

    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump({"best_model": best_name, "accuracies": results}, f, indent=2)

    print(f"\nSaved model.pkl, vectorizer.pkl, metadata.json to {MODEL_DIR}/")


if __name__ == "__main__":
    main()