import pytest
import numpy as np
from src.models.semantic import extract_change_regions

def test_extract_change_regions_filtering():
    """Verify regions smaller than min_size are filtered out correctly."""
    h, w = 100, 100
    semantic_change_mask = np.zeros((h, w), dtype=np.int32)
    
    # Region 1: Road (Class 1) of area 20 pixels
    semantic_change_mask[10:15, 10:14] = 1 # 5 * 4 = 20 px
    
    # Region 2: Building (Class 2) of area 5 pixels (below min_size=10)
    semantic_change_mask[30:35, 30:31] = 2 # 5 * 1 = 5 px

    regions = extract_change_regions(semantic_change_mask, min_size=10)

    # Should only keep Region 1
    assert len(regions) == 1
    r = regions[0]
    assert r["Class ID"] == 1
    assert r["Class"] == "road"
    assert r["Area"] == 20
    assert r["X"] == 10
    assert r["Y"] == 10
    assert r["Width"] == 4
    assert r["Height"] == 5
    assert 0 <= r["Centroid X"] < w
    assert 0 <= r["Centroid Y"] < h

def test_region_bounding_box_within_image():
    """Verify all bounding box coordinates stay strictly within image dimensions."""
    h, w = 50, 50
    mask = np.ones((h, w), dtype=np.int32) # Entire grid is Class 1
    regions = extract_change_regions(mask, min_size=10)

    assert len(regions) == 1
    r = regions[0]
    assert r["X"] >= 0 and (r["X"] + r["Width"]) <= w
    assert r["Y"] >= 0 and (r["Y"] + r["Height"]) <= h
