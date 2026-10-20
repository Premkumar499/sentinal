import pytest
from pathlib import Path
from src.temporal_fetcher import TemporalFetcher, lon_lat_to_tile
from src.inference import InferenceEngine
from src.search_engine import RegionDatabase


def test_tile_coordinate_math():
    # Chennai bounds
    lon, lat = 80.090332, 13.261333
    x, y = lon_lat_to_tile(lon, lat, 14)
    assert x == 11836 or x == 11837
    assert y == 7582 or y == 7583


def test_tamilnadu_database_indexed():
    db = RegionDatabase()
    results = db.query_regions(limit=50)
    # Ensure regions are present
    assert len(results) > 0
    # Check that Tamil Nadu regions have valid coordinates
    tn_regions = [r for r in results if "_CLOSE_" in str(r.get("scene_id", ""))]
    if tn_regions:
        sample = tn_regions[0]
        assert sample.get("centroid_lat") is not None
        assert sample.get("centroid_lon") is not None
        assert sample.get("area_m2") is not None
