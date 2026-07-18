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
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "products_clean.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "recommender.joblib"

MAX_NEIGHBORS = 25  # precompute enough neighbors to serve any k <= this at query time


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    import torch
    # Limit CPU thread contention which slows down PyTorch on many-core processors
    torch.set_num_threads(4)
    
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["uniq_id"])
    df["description"] = df["description"].fillna("")
    df["product_name"] = df["product_name"].fillna("")

    # Truncate description and combine with name to keep sequence lengths small
    text = df["product_name"] + " " + df["description"].str.slice(0, 300)

    logger.info("Loading sentence-transformers model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Limit max sequence length to 128 tokens to speed up self-attention and reduce padding overhead
    model.max_seq_length = 128
    
    logger.info("Generating embeddings for %d products...", len(df))
    embeddings = model.encode(text.tolist(), show_progress_bar=True, batch_size=256)
    logger.info("Embeddings shape: %s", embeddings.shape)

    n_neighbors = min(MAX_NEIGHBORS + 1, embeddings.shape[0])  # +1 because a product is its own nearest neighbor
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn_index.fit(embeddings)
    logger.info("Fitted NearestNeighbors with n_neighbors=%d", n_neighbors)

    # Precompute neighbors for every existing product so "similar to product X"
    # lookups are an O(1) array index at request time instead of re-running
    # kneighbors over the full matrix on every API call.
    logger.info("Precomputing neighbor table for all %d products...", embeddings.shape[0])
    distances, indices = nn_index.kneighbors(embeddings, n_neighbors=n_neighbors)

    artifact = {
        "embeddings": embeddings,
        "nn_index": nn_index,
        "uniq_ids": df["uniq_id"].tolist(),
        "precomputed_distances": distances,
        "precomputed_indices": indices,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    logger.info("Saved recommender artifact to %s", model_path)

    return {"n_products": embeddings.shape[0], "embedding_dim": embeddings.shape[1]}


if __name__ == "__main__":
    metrics = train()
    print(metrics)
