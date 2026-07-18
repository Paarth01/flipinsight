"""
Train a retail-price prediction model.

Features: category_l1, category_l2, brand, description_length,
          description_word_count, spec_count, is_FK_Advantage_product
Target:   retail_price (log-transformed to tame the heavy right skew)
Model:    RandomForestRegressor inside an sklearn Pipeline (OneHot + impute)
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "products_clean.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "price_model.joblib"

CATEGORICAL_FEATURES = ["category_l1", "category_l2", "brand"]
NUMERIC_FEATURES = ["description_length", "description_word_count", "spec_count", "is_FK_Advantage_product"]
TARGET = "retail_price"


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
            ]), CATEGORICAL_FEATURES),
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ]
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = np.log1p(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    logger.info("Training on %d rows", len(X_train))
    pipeline.fit(X_train, y_train)

    preds_log = pipeline.predict(X_test)
    preds = np.expm1(preds_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)
    logger.info("Test MAE: Rs. %.2f | R2: %.3f", mae, r2)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)

    return {"mae": mae, "r2": r2, "n_train": len(X_train), "n_test": len(X_test)}


if __name__ == "__main__":
    metrics = train()
    print(metrics)
