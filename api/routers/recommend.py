"""Similar-product recommendation endpoints (content-based, TF-IDF + cosine similarity)."""

import logging
from pathlib import Path

import joblib
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.schemas import RecommendBySearchRequest, RecommendResponse, SimilarProduct
from db.models import Product, get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommend", tags=["recommend"])

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "models" / "recommender.joblib"

_artifact = None


def _get_artifact():
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Recommender not trained yet. Run ml/train_recommender.py")
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def _hydrate(uniq_ids: list[str], similarities: list[float], db: Session) -> list[SimilarProduct]:
    """Fetch full product rows for a list of uniq_ids, preserving order, and attach similarity scores."""
    if not uniq_ids:
        return []
    products = db.query(Product).filter(Product.uniq_id.in_(uniq_ids)).all()
    by_id = {p.uniq_id: p for p in products}
    results = []
    for uid, sim in zip(uniq_ids, similarities):
        p = by_id.get(uid)
        if p is None:
            continue
        results.append(SimilarProduct(
            uniq_id=p.uniq_id,
            product_name=p.product_name,
            brand=p.brand,
            category_l1=p.category_l1,
            retail_price=p.retail_price,
            discounted_price=p.discounted_price,
            product_url=p.product_url,
            similarity=round(sim, 4),
        ))
    return results


@router.get("/similar/{uniq_id}", response_model=RecommendResponse)
def similar_to_product(uniq_id: str, k: int = Query(10, ge=1, le=25), db: Session = Depends(get_db)):
    artifact = _get_artifact()
    uniq_ids = artifact["uniq_ids"]
    if uniq_id not in uniq_ids:
        raise HTTPException(status_code=404, detail="Product not found in recommender index")

    row_idx = uniq_ids.index(uniq_id)
    distances = artifact["precomputed_distances"][row_idx]
    indices = artifact["precomputed_indices"][row_idx]

    # the product itself is always its own nearest neighbor at distance ~0; drop it
    neighbor_uids, sims = [], []
    for dist, idx in zip(distances, indices):
        if uniq_ids[idx] == uniq_id:
            continue
        neighbor_uids.append(uniq_ids[idx])
        sims.append(1 - dist)  # cosine distance -> cosine similarity
        if len(neighbor_uids) == k:
            break

    results = _hydrate(neighbor_uids, sims, db)
    return RecommendResponse(source=uniq_id, results=results)


@router.post("/search", response_model=RecommendResponse)
def similar_by_text(payload: RecommendBySearchRequest, db: Session = Depends(get_db)):
    artifact = _get_artifact()
    vectorizer = artifact["vectorizer"]
    nn_index = artifact["nn_index"]
    uniq_ids = artifact["uniq_ids"]

    query_vec = vectorizer.transform([payload.query])
    n_neighbors = min(payload.k, len(uniq_ids))
    distances, indices = nn_index.kneighbors(query_vec, n_neighbors=n_neighbors)

    neighbor_uids = [uniq_ids[i] for i in indices[0]]
    sims = [1 - d for d in distances[0]]

    results = _hydrate(neighbor_uids, sims, db)
    return RecommendResponse(source=payload.query, results=results)
