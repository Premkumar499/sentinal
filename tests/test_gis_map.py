import pytest
from app.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_map_page_renders_leaflet(client):
    """Verify that /map renders the Leaflet GIS map container."""
    response = client.get("/map")
    assert response.status_code == 200
    assert b"id=\"leaflet_map\"" in response.data
    assert b"leaflet.js" in response.data
    assert b"districtCenters" in response.data

def test_api_gis_regions(client):
    """Verify that /api/gis/regions returns valid georeferenced satellite regions."""
    response = client.get("/api/gis/regions")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["total"] > 0
    assert isinstance(data["regions"], list)

    first_region = data["regions"][0]
    assert "lat" in first_region
    assert "lon" in first_region
    assert "area_m2" in first_region
    assert "semantic_class" in first_region
    assert "district" in first_region
    assert "crop_url" in first_region
    assert first_region["lat"] != 0

def test_api_gis_regions_filtering(client):
    """Verify district and semantic class filtering."""
    # Test district filter
    res_chennai = client.get("/api/gis/regions?district=Chennai")
    assert res_chennai.status_code == 200
    data_chennai = res_chennai.get_json()
    for r in data_chennai["regions"]:
        assert "chennai" in r["district"].lower()

    # Test class filter
    res_water = client.get("/api/gis/regions?semantic_class=water")
    assert res_water.status_code == 200
    data_water = res_water.get_json()
    for r in data_water["regions"]:
        assert r["semantic_class"].lower() == "water"

def test_api_gis_districts_geojson(client):
    """Verify that /api/gis/districts-geojson serves valid GeoJSON."""
    response = client.get("/api/gis/districts-geojson")
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("type") == "FeatureCollection"
    assert "features" in data
