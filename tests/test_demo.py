import pytest
from app.app import app

def test_demo_presentation_flow():
    client = app.test_client()

    # 1. Homepage
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert b"Semantic Satellite Change Detection" in res_home.data

    # 2. Demo Execution Route (Follow redirect to /result)
    res_demo = client.get("/demo?scene=test_006", follow_redirects=True)
    assert res_demo.status_code == 200
    assert b"Analysis Completed" in res_demo.data or b"Results" in res_demo.data or b"Semantic" in res_demo.data

    # 3. Search Changes
    res_search = client.get("/search?q=road")
    assert res_search.status_code == 200
    assert b"road" in res_search.data.lower()

    # 4. Map View
    res_map = client.get("/map")
    assert res_map.status_code == 200

    # 5. Model Explainability Page
    res_model = client.get("/model")
    assert res_model.status_code == 200
    assert b"Model Audit" in res_model.data

    # 6. Reset Demo State
    res_reset = client.post("/api/demo/reset")
    assert res_reset.status_code == 200
    assert res_reset.json["status"] == "success"
