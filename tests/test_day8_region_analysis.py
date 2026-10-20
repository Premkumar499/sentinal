import pytest
import numpy as np
from src.models.semantic import (
    determine_change_type,
    calculate_semantic_transition_matrix,
    extract_change_regions
)

def test_determine_change_type():
    """Verify change-type categorization logic."""
    assert determine_change_type(0, 1) == "new"       # background -> road
    assert determine_change_type(2, 0) == "removed"   # building -> background
    assert determine_change_type(1, 1) == "expanded"  # road -> road
    assert determine_change_type(4, 2) == "transition"# vegetation -> building
    assert determine_change_type(0, 0) == "unknown"   # unknown -> unknown

def test_semantic_transition_matrix_synthetic():
    """Verify transition matrix pixel count calculations."""
    h, w = 10, 10
    t1_mask = np.zeros((h, w), dtype=np.int32)
    t2_mask = np.zeros((h, w), dtype=np.int32)
    change_mask = np.zeros((h, w), dtype=np.int32)

    # 4 pixels background -> road
    t2_mask[0:2, 0:2] = 1
    change_mask[0:2, 0:2] = 1

    # 2 pixels building -> background
    t1_mask[5:7, 5:6] = 2
    change_mask[5:7, 5:6] = 1

    matrix = calculate_semantic_transition_matrix(t1_mask, t2_mask, change_mask)
    assert len(matrix) >= 2
    
    road_trans = [m for m in matrix if m["to_class"] == "road"][0]
    assert road_trans["pixel_count"] == 4
    assert road_trans["from_class"] == "background"

def test_extract_change_regions_day8_ranking_and_flags():
    """Verify Day 8 region ranking, significance score, and quality flags."""
    h, w = 50, 50
    sem_change_mask = np.zeros((h, w), dtype=np.int32)
    t1_mask = np.zeros((h, w), dtype=np.int32)
    t2_mask = np.zeros((h, w), dtype=np.int32)
    prob_map = np.ones((h, w), dtype=np.float32) * 0.90

    # Region 1: Large Road region (Area = 100 px)
    sem_change_mask[10:20, 10:20] = 1
    t2_mask[10:20, 10:20] = 1

    # Region 2: Small Building region (Area = 16 px)
    sem_change_mask[30:34, 30:34] = 2
    t2_mask[30:34, 30:34] = 2

    regions = extract_change_regions(
        semantic_change_mask=sem_change_mask,
        min_size=10,
        t1_mask=t1_mask,
        t2_mask=t2_mask,
        prob_map=prob_map
    )

    assert len(regions) == 2
    # Verify ranking (highest significance score first)
    assert regions[0]["Rank"] == 1
    assert regions[0]["Class"] == "road"
    assert regions[0]["Area"] == 100
    assert regions[0]["Change Type"] == "new"
    assert regions[0]["Significance Score"] > regions[1]["Significance Score"]
    assert "alignment_ok" in regions[0]["Quality Flags"]
