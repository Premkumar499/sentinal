import pytest
import numpy as np
from src.models.semantic import compute_temporal_semantic_change, get_semantic_mask, detect_semantic_change

def test_semantic_change_matrix_synthetic():
    """
    Test exact element-wise intersection of change_mask and semantic_mask.
    """
    change_mask = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0]
    ], dtype=np.int32)

    semantic_mask_t2 = np.array([
        [0, 1, 1, 2],
        [0, 2, 1, 0]
    ], dtype=np.int32)

    semantic_mask_t1 = np.zeros_like(semantic_mask_t2)

    expected_sem_change = np.array([
        [0, 0, 1, 2],
        [0, 2, 1, 0]
    ], dtype=np.int32)

    actual_sem_change = compute_temporal_semantic_change(change_mask, semantic_mask_t1, semantic_mask_t2)

    np.testing.assert_array_equal(actual_sem_change, expected_sem_change)

def test_semantic_mask_classes_range():
    """Verify class IDs fall strictly in range 0..6."""
    bands_norm = {k: np.random.rand(20, 20).astype(np.float32) for k in ["red", "green", "blue", "nir", "swir"]}
    mask = get_semantic_mask(bands_norm, source_type="spectral_rules")
    unique_ids = set(np.unique(mask))
    assert unique_ids.issubset({0, 1, 2, 3, 4, 5, 6})
