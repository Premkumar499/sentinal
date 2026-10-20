import pytest
import sqlite3
from pathlib import Path
from src.search_engine import RegionDatabase, QueryParser
from app.app import app

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_regions.db"
    return RegionDatabase(db_path=db_file)

def test_query_parser_patterns():
    # 1. Road query
    res = QueryParser.parse_query("show road changes")
    assert res["status"] == "success"
    assert res["intent"]["semantic_class"] == "road"

    # 2. Building query
    res = QueryParser.parse_query("find building changes")
    assert res["status"] == "success"
    assert res["intent"]["semantic_class"] == "building"

    # 3. Water query
    res = QueryParser.parse_query("show water changes")
    assert res["status"] == "success"
    assert res["intent"]["semantic_class"] == "water"

    # 4. Large area query
    res = QueryParser.parse_query("show large changes")
    assert res["status"] == "success"
    assert res["intent"]["min_area"] == 200
    assert res["intent"]["sort_by"] == "area_desc"

    # 5. Combined query: New road regions
    res = QueryParser.parse_query("find new road regions")
    assert res["status"] == "success"
    assert res["intent"]["semantic_class"] == "road"
    assert res["intent"]["change_type"] == "new"

    # 6. Unsupported class handling
    res = QueryParser.parse_query("find airport changes")
    assert res["status"] == "unsupported"
    assert "Airport" in res["message"]

    # 7. Empty query
    res = QueryParser.parse_query("   ")
    assert res["status"] == "empty"

def test_sqlite_region_database(temp_db):
    sample_regions = [
        {
            "Region ID": 1,
            "Class": "road",
            "Change Type": "new",
            "Area": 500,
            "Significance Score": 0.85,
            "Bounding Box [X,Y,W,H]": [10, 20, 30, 40],
            "Centroid [CX,CY]": [25, 40],
            "T1 Class": "bare_land",
            "T2 Class": "road"
        },
        {
            "Region ID": 2,
            "Class": "building",
            "Change Type": "expanded",
            "Area": 1200,
            "Significance Score": 0.92,
            "Bounding Box [X,Y,W,H]": [50, 60, 70, 80],
            "Centroid [CX,CY]": [85, 100],
            "T1 Class": "building",
            "T2 Class": "building"
        }
    ]

    temp_db.insert_regions(run_id="run_test", scene_id="scene_01", regions=sample_regions)

    # Query Road
    roads = temp_db.query_regions(semantic_class="road")
    assert len(roads) == 1
    assert roads[0]["semantic_class"] == "road"
    assert roads[0]["area_pixels"] == 500

    # Query Building
    buildings = temp_db.query_regions(semantic_class="building")
    assert len(buildings) == 1
    assert buildings[0]["semantic_class"] == "building"
    assert buildings[0]["area_pixels"] == 1200

    # Query All sorted by area desc
    all_regions = temp_db.query_regions(sort_by="area_desc")
    assert len(all_regions) == 2
    assert all_regions[0]["area_pixels"] == 1200

def test_flask_search_api_endpoints():
    client = app.test_client()

    # 1. Natural Language API Search
    res = client.post("/api/search", json={"query": "show road changes"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "intent" in data
    assert "summary" in data
    assert "results" in data

    # 2. Structured Filter API Search
    res_filters = client.post("/api/search/filters", json={"semantic_class": "building", "min_area": 10})
    assert res_filters.status_code == 200
    data_f = res_filters.get_json()
    assert data_f["status"] == "success"

    # 3. Search CSV Export
    res_csv = client.get("/api/search/export?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["Content-Type"]

    # 4. Search Page UI Render
    res_ui = client.get("/search")
    assert res_ui.status_code == 200
    assert b"Search & Interpret Detected Changes" in res_ui.data

def test_exact_location_queries():
    """Verify natural language geographic queries (e.g. 'road change in karur') resolve exact GPS coordinates."""
    # 1. Test QueryParser geographic extraction
    parsed = QueryParser.parse_query("road change in karur")
    assert parsed["status"] == "success"
    assert parsed["intent"]["district"] == "karur"
    assert parsed["intent"]["semantic_class"] == "road"

    # 2. Test API search returns exact physical location object
    client = app.test_client()
    res = client.post("/api/search", json={"query": "road change in karur"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["intent"]["district"] == "karur"
    assert len(data["results"]) > 0

    first_res = data["results"][0]
    assert "exact_location" in first_res
    loc = first_res["exact_location"]
    assert loc is not None
    assert loc["district"].lower() == "karur"
    assert "latitude" in loc and isinstance(loc["latitude"], float)
    assert "longitude" in loc and isinstance(loc["longitude"], float)
    assert "map_url" in loc and "/map?lat=" in loc["map_url"]
    assert "google_maps_url" in loc and "google.com/maps" in loc["google_maps_url"]

    # 3. Test water change in chennai
    res_chennai = client.post("/api/search", json={"query": "water change in chennai"})
    assert res_chennai.status_code == 200
    data_c = res_chennai.get_json()
    assert data_c["intent"]["district"] == "chennai"
    assert len(data_c["results"]) > 0
    loc_c = data_c["results"][0]["exact_location"]
    assert loc_c["district"].lower() == "chennai"
    assert "° N" in loc_c["coordinates_str"]
