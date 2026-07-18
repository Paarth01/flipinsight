def test_recommend_search():
    from tests.test_api import client
    r = client.post(
        "/recommend/search",
        json={"query": "waterproof running shoes for men", "k": 5},
        headers={"X-API-Key": "dev-secret-key"}
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 5
    assert all("similarity" in item for item in data["results"])


def test_recommend_similar_by_id():
    from tests.test_api import client
    listing = client.get("/products?page_size=1").json()
    uid = listing["items"][0]["uniq_id"]
    r = client.get(f"/recommend/similar/{uid}?k=5", headers={"X-API-Key": "dev-secret-key"})
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == uid
    assert len(data["results"]) <= 5
    # the product should not recommend itself
    assert all(item["uniq_id"] != uid for item in data["results"])


def test_recommend_similar_not_found():
    from tests.test_api import client
    r = client.get("/recommend/similar/nonexistent-id-999?k=5", headers={"X-API-Key": "dev-secret-key"})
    assert r.status_code == 404


def test_recommend_search_k_bounds():
    from tests.test_api import client
    r = client.post(
        "/recommend/search",
        json={"query": "test", "k": 100},
        headers={"X-API-Key": "dev-secret-key"}
    )
    assert r.status_code == 422  # k must be <= 25
