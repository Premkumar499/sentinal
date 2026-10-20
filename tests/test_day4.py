import numpy as np
import pytest
import json
from pathlib import Path
from src.models.semantic import (
    get_placeholder_semantic_mask,
    get_semantic_mask,
    compute_semantic_change,
    compute_road_change,
    compute_temporal_semantic_change,
    extract_change_regions,
    detect_semantic_change
)

def test_semantic_classes_json():
    path = Path("config/semantic_classes.json")
    assert path.exists(), "semantic_classes.json does not exist"
    
    with open(path, "r") as f:
        data = json.load(f)
        
    assert "0" in data
    assert data["0"] == "background"
    assert "1" in data
    assert data["1"] == "road"
    assert "2" in data
    assert data["2"] == "building"

def test_get_semantic_mask_placeholder():
    shape = (50, 50)
    # Verify placeholder works
    mask = get_placeholder_semantic_mask(shape, "road")
    assert mask.shape == shape
    assert np.any(mask == 1)

def test_get_semantic_mask_spectral_rules():
    shape = (100, 100)
    # Generate mock normalized bands
    # Water: green - nir > 0.05
    green = np.zeros(shape, dtype=np.float32)
    green[:, :20] = 0.5
    nir = np.zeros(shape, dtype=np.float32)
    nir[:, :20] = 0.1 # NDWI = (0.5-0.1)/(0.6) = 0.66 > 0.05
    
    # Vegetation: nir - red > 0.25
    red = np.zeros(shape, dtype=np.float32)
    red[:, 20:40] = 0.1
    nir[:, 20:40] = 0.6 # NDVI = (0.6-0.1)/0.7 = 0.71 > 0.25
    
    # Built-up: swir - nir > 0.05, NDVI <= 0.25
    red = np.zeros(shape, dtype=np.float32)
    red[:, 20:40] = 0.1
    red[:, 40:70] = 0.3
    nir[:, 40:70] = 0.2 # NDBI = (0.6-0.2)/0.8 = 0.5 > 0.05, NDVI = (0.2-0.3)/0.5 = -0.2 < 0.25
    swir = np.zeros(shape, dtype=np.float32)
    swir[:, 40:70] = 0.6
    
    blue = np.zeros(shape, dtype=np.float32)
    
    bands = {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "swir": swir
    }
    
    mask = get_semantic_mask(bands, source_type="spectral_rules")
    
    assert mask.shape == shape
    # Verify water pixels mapped to 3
    assert np.all(mask[:, :20] == 3)
    # Verify vegetation pixels mapped to 4
    assert np.all(mask[:, 20:40] == 4)
    # Verify built-up features mapped to either 1 (road) or 2 (building)
    assert np.all(np.isin(mask[:, 40:70], [1, 2]))

def test_compute_temporal_semantic_change():
    shape = (50, 50)
    change_mask = np.zeros(shape, dtype=np.int32)
    change_mask[10:30, 10:30] = 1
    
    t1_mask = np.zeros(shape, dtype=np.int32)
    t2_mask = np.zeros(shape, dtype=np.int32)
    
    # Set building change
    t2_mask[10:20, 10:20] = 2 # building class
    # Set water change
    t2_mask[20:30, 20:30] = 3 # water class
    
    sem_change = compute_temporal_semantic_change(change_mask, t1_mask, t2_mask)
    
    assert sem_change.shape == shape
    # Outside change mask must be 0
    assert np.all(sem_change[0:5, :] == 0)
    # Inside change mask matches T2
    assert np.all(sem_change[10:20, 10:20] == 2)
    assert np.all(sem_change[20:30, 20:30] == 3)

def test_extract_change_regions():
    shape = (50, 50)
    sem_change = np.zeros(shape, dtype=np.int32)
    # Create two isolated building change blocks (class 2)
    sem_change[5:15, 5:15] = 2
    sem_change[30:40, 30:40] = 2
    
    regions = extract_change_regions(sem_change, min_size=5)
    
    assert len(regions) == 2
    assert regions[0]["Class"] == "building"
    assert regions[1]["Class"] == "building"
    assert regions[0]["Area"] == 100
    assert regions[1]["Area"] == 100
    # check bounds
    assert regions[0]["Width"] == 10
    assert regions[0]["Height"] == 10

def test_detect_semantic_change_api():
    shape = (100, 100)
    change_mask = np.zeros(shape, dtype=np.int32)
    change_mask[40:60, 40:60] = 1
    
    t1_bands = {k: np.zeros(shape, dtype=np.float32) for k in ["blue", "green", "red", "nir", "swir"]}
    t2_bands = {k: np.zeros(shape, dtype=np.float32) for k in ["blue", "green", "red", "nir", "swir"]}
    
    # Mock built-up in T2
    t2_bands["swir"][40:60, 40:60] = 0.5
    t2_bands["nir"][40:60, 40:60] = 0.1
    t2_bands["red"][40:60, 40:60] = 0.2
    
    res = detect_semantic_change(t1_bands, t2_bands, change_mask, semantic_source="spectral_rules")
    
    assert "change_mask" in res
    assert "semantic_change_mask" in res
    assert "regions" in res
    assert "statistics" in res
    
    stats = res["statistics"]
    assert stats["total_changed_pixels"] == 400
    assert stats["building_pixels"] + stats["road_pixels"] + stats["unknown_pixels"] == 400
