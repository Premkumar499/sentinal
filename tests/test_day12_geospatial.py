import pytest
import numpy as np
from pathlib import Path
from src.geospatial import (
    get_geospatial_metadata,
    pixel_to_geo,
    geo_to_pixel,
    calculate_region_area_m2,
    generate_region_geojson,
    draw_region_boundaries_image
)
from app.app import app

def test_get_geospatial_metadata_unreferenced(tmp_path):
    # Test fallback metadata for standard PNG/JPG or unreferenced raster
    dummy_img = tmp_path / "dummy.png"
    import cv2
    cv2.imwrite(str(dummy_img), np.zeros((100, 200, 3), dtype=np.uint8))

    meta = get_geospatial_metadata(str(dummy_img))
    assert meta["width"] == 200
    assert meta["height"] == 100
    assert meta["coordinate_type"] == "pixel"
    assert meta["resolution_m"] == [10.0, 10.0]

def test_calculate_region_area_m2():
    meta = {"resolution_m": [10.0, 10.0], "is_georeferenced": False}
    area_res = calculate_region_area_m2(area_pixels=500, metadata=meta)
    assert area_res["area_pixels"] == 500
    assert area_res["area_m2"] == 50000.0 # 500 * 100
    assert area_res["area_hectares"] == 5.0 # 50000 / 10000

def test_draw_region_boundaries_image(tmp_path):
    output_png = tmp_path / "boundaries.png"
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    sample_regions = [
        {"Region ID": 1, "Class": "road", "Bounding Box [X,Y,W,H]": [10, 20, 30, 40]}
    ]
    draw_region_boundaries_image(rgb, sample_regions, str(output_png))
    assert output_png.exists()

def test_generate_region_geojson_unreferenced():
    meta = {"is_georeferenced": False}
    sample_regions = [{"Region ID": 1, "Class": "road", "Bounding Box [X,Y,W,H]": [10, 20, 30, 40]}]
    geojson = generate_region_geojson("scene_01", sample_regions, meta)
    # Should safely return None for unreferenced rasters (no fake GPS coordinates!)
    assert geojson is None

def test_flask_map_and_region_api_endpoints():
    client = app.test_client()

    # 1. Map View Page
    res_map = client.get("/map")
    assert res_map.status_code == 200
    assert b"Region Boundary & Spatial Location Map" in res_map.data

    # 2. GET Region Details API (404 for nonexistent)
    res_reg = client.get("/api/regions/invalid_run/999")
    assert res_reg.status_code == 404

    # 3. GET GeoJSON API (404 for unreferenced)
    res_geo = client.get("/api/regions/test_000/geojson")
    assert res_geo.status_code == 404
