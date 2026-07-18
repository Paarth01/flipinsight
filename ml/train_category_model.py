"""
Train a text classifier that predicts category_l1 from the product description.

Model: TF-IDF (unigrams+bigrams) -> LogisticRegression
Only categories with >= MIN_CATEGORY_COUNT examples are kept, to avoid
single-example classes that break stratified splitting and add noise.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "products_clean.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "category_model.joblib"
MIN_CATEGORY_COUNT = 20


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=-1, C=5.0)),
    ])


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["description", "category_l1"])
    df = df[df["description"].str.strip() != ""]

    counts = df["category_l1"].value_counts()
    keep_categories = counts[counts >= MIN_CATEGORY_COUNT].index
    df = df[df["category_l1"].isin(keep_categories)]
    logger.info("Training on %d categories, %d rows after filtering rare classes", df["category_l1"].nunique(), len(df))

    X = df["description"]
    y = df["category_l1"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info("Test accuracy: %.4f", acc)
    report = classification_report(y_test, preds, zero_division=0)
    logger.info("\n%s", report)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    return {"accuracy": acc, "n_train": len(X_train), "n_test": len(X_test), "n_classes": df["category_l1"].nunique()}


if __name__ == "__main__":
    metrics = train()
    print(metrics)
