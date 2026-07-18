"""
Train a similar-product recommender using TF-IDF over product_name + description,
with cosine-similarity nearest-neighbor lookup.

This is content-based (no user interaction data exists in this dataset), which
is the right fit here: given a product (or a free-text query), find the most
textually/semantically similar listings by name + description.

Artifact saved: {vectorizer, nn_index, uniq_ids, matrix}
  - vectorizer: fitted TfidfVectorizer
  - nn_index:   fitted sklearn NearestNeighbors (cosine metric)
  - uniq_ids:   list mapping matrix row -> product uniq_id, for DB lookups
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "products_clean.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "recommender.joblib"

MAX_NEIGHBORS = 25  # precompute enough neighbors to serve any k <= this at query time


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["uniq_id"])
    df["description"] = df["description"].fillna("")
    df["product_name"] = df["product_name"].fillna("")

    # Weight the product name more heavily than description by repeating it —
    # cheap trick that noticeably improves relevance for short-name matches.
    text = (df["product_name"] + " ") * 3 + df["description"]

    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, stop_words="english")
    matrix = vectorizer.fit_transform(text)
    logger.info("TF-IDF matrix shape: %s", matrix.shape)

    n_neighbors = min(MAX_NEIGHBORS + 1, matrix.shape[0])  # +1 because a product is its own nearest neighbor
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn_index.fit(matrix)
    logger.info("Fitted NearestNeighbors with n_neighbors=%d", n_neighbors)

    # Precompute neighbors for every existing product so "similar to product X"
    # lookups are an O(1) array index at request time instead of re-running
    # kneighbors over the full matrix on every API call.
    logger.info("Precomputing neighbor table for all %d products...", matrix.shape[0])
    distances, indices = nn_index.kneighbors(matrix, n_neighbors=n_neighbors)

    artifact = {
        "vectorizer": vectorizer,
        "nn_index": nn_index,
        "uniq_ids": df["uniq_id"].tolist(),
        "precomputed_distances": distances,
        "precomputed_indices": indices,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    logger.info("Saved recommender artifact to %s", model_path)

    return {"n_products": matrix.shape[0], "vocab_size": len(vectorizer.vocabulary_)}


if __name__ == "__main__":
    metrics = train()
    print(metrics)
