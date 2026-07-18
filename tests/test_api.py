"""Basic tests for FlipInsight API endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_serves_dashboard():
    r = client.get("/")
    assert r.status_code == 200
    assert "FlipInsight" in r.text


def test_summary_stats():
    r = client.get("/analytics/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["total_products"] > 0
    assert data["total_brands"] > 0
    assert data["avg_retail_price"] > 0


def test_category_counts():
    r = client.get("/analytics/categories?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data) <= 5
    assert data[0]["count"] >= data[-1]["count"]


def test_price_by_category():
    r = client.get("/analytics/price-by-category?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert all("avg_retail_price" in row for row in data)


def test_top_brands():
    r = client.get("/analytics/top-brands?limit=5")
    assert r.status_code == 200


def test_discount_distribution():
    r = client.get("/analytics/discount-distribution")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 6


def test_list_products_default():
    r = client.get("/products")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] > 0
    assert len(data["items"]) == 20


def test_list_products_filter_category():
    r = client.get("/products?category=Clothing&page_size=5")
    assert r.status_code == 200
    data = r.json()
    assert all(item["category_l1"] == "Clothing" for item in data["items"])


def test_list_products_search():
    r = client.get("/products?search=shoes&page_size=5")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 0


def test_get_product_not_found():
    r = client.get("/products/nonexistent-id-123")
    assert r.status_code == 404


def test_get_product_valid():
    listing = client.get("/products?page_size=1").json()
    uid = listing["items"][0]["uniq_id"]
    r = client.get(f"/products/{uid}")
    assert r.status_code == 200
    assert r.json()["uniq_id"] == uid


def test_predict_price():
    r = client.post("/predict/price", json={
        "category_l1": "Clothing",
        "category_l2": "Women's Clothing",
        "brand": "Alisha",
        "description": "Cotton casual shorts for women, comfortable fit",
        "is_FK_Advantage_product": False,
    }, headers={"X-API-Key": "dev-secret-key"})
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_retail_price"] > 0


def test_predict_category():
    r = client.post("/predict/category", json={
        "description": "Men's running shoes with breathable mesh upper and cushioned sole"
    }, headers={"X-API-Key": "dev-secret-key"})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_category" in data
    assert len(data["top_3"]) == 3


def test_predict_price_missing_required_field():
    r = client.post("/predict/price", json={"description": "test"}, headers={"X-API-Key": "dev-secret-key"})
    assert r.status_code == 422


def test_predict_unauthorized_no_key():
    r = client.post("/predict/price", json={"description": "test"})
    assert r.status_code == 401


def test_predict_unauthorized_bad_key():
    r = client.post("/predict/price", json={"description": "test"}, headers={"X-API-Key": "bad-key"})
    assert r.status_code == 401


def test_predict_rate_limiter():
    # Make multiple requests to trigger 429
    # Predict router allows 5 requests per 60 seconds.
    # The first 5 requests (including the successful test_predict_price/test_predict_price_missing_required_field above)
    # might have consumed some quota. To be safe and deterministic, let's clear the history or just flood it.
    from api.security import _request_history
    _request_history.clear()

    for _ in range(5):
        r = client.post("/predict/price", json={
            "category_l1": "Clothing",
            "description": "test"
        }, headers={"X-API-Key": "dev-secret-key"})
        assert r.status_code in (200, 422)  # missing category_l2/brand is fine, might return 200 or 422 depending on schema
    
    # The 6th request must be rate limited
    r = client.post("/predict/price", json={
        "category_l1": "Clothing",
        "description": "test"
    }, headers={"X-API-Key": "dev-secret-key"})
    assert r.status_code == 429
