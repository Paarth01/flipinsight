"""ML inference endpoints — price prediction and category classification."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    CategoryPredictionRequest,
    CategoryPredictionResponse,
    PricePredictionRequest,
    PricePredictionResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["ml"])

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "models"

_price_model = None
_category_model = None


def _get_price_model():
    global _price_model
    if _price_model is None:
        path = MODEL_DIR / "price_model.joblib"
        if not path.exists():
            raise HTTPException(status_code=503, detail="Price model not trained yet. Run ml/train_price_model.py")
        _price_model = joblib.load(path)
    return _price_model


def _get_category_model():
    global _category_model
    if _category_model is None:
        path = MODEL_DIR / "category_model.joblib"
        if not path.exists():
            raise HTTPException(status_code=503, detail="Category model not trained yet. Run ml/train_category_model.py")
        _category_model = joblib.load(path)
    return _category_model


@router.post("/price", response_model=PricePredictionResponse)
def predict_price(payload: PricePredictionRequest):
    model = _get_price_model()
    row = pd.DataFrame([{
        "category_l1": payload.category_l1,
        "category_l2": payload.category_l2,
        "brand": payload.brand,
        "description_length": len(payload.description),
        "description_word_count": len(payload.description.split()),
        "spec_count": 0,
        "is_FK_Advantage_product": payload.is_FK_Advantage_product,
    }])
    pred_log = model.predict(row)[0]
    pred = float(np.expm1(pred_log))
    return PricePredictionResponse(predicted_retail_price=round(pred, 2))


@router.post("/category", response_model=CategoryPredictionResponse)
def predict_category(payload: CategoryPredictionRequest):
    model = _get_category_model()
    proba = model.predict_proba([payload.description])[0]
    classes = model.classes_
    top_idx = np.argsort(proba)[::-1][:3]
    top_3 = [{"category": classes[i], "confidence": round(float(proba[i]), 4)} for i in top_idx]
    return CategoryPredictionResponse(
        predicted_category=classes[top_idx[0]],
        confidence=round(float(proba[top_idx[0]]), 4),
        top_3=top_3,
    )
