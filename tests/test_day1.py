import numpy as np
import pytest
from src.features.extraction import calculate_ndvi, calculate_ndbi, FeatureExtractor
from src.preprocessing.pipeline import PreprocessingPipeline

def test_calculate_ndvi():
    nir = np.array([[0.8, 0.2], [0.0, 0.5]], dtype=float)
    red = np.array([[0.2, 0.2], [0.0, 0.5]], dtype=float)
    
    ndvi = calculate_ndvi(nir, red)
    
    # Check expected outputs
    # (0.8 - 0.2) / (0.8 + 0.2) = 0.6 / 1.0 = 0.6
    # (0.2 - 0.2) / (0.2 + 0.2) = 0.0
    # (0 - 0) / (0 + 0) = 0.0 (safe zero handling)
    assert np.isclose(ndvi[0, 0], 0.6)
    assert np.isclose(ndvi[0, 1], 0.0)
    assert np.isclose(ndvi[1, 0], 0.0)
    assert np.isclose(ndvi[1, 1], 0.0)

def test_calculate_ndbi():
    swir = np.array([[0.5, 0.1], [0.0, 0.3]], dtype=float)
    nir = np.array([[0.1, 0.3], [0.0, 0.3]], dtype=float)
    
    ndbi = calculate_ndbi(swir, nir)
    
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.666667
    # (0.1 - 0.3) / (0.1 + 0.3) = -0.2 / 0.4 = -0.5
    assert np.isclose(ndbi[0, 0], 0.4 / 0.6)
    assert np.isclose(ndbi[0, 1], -0.5)
    assert np.isclose(ndbi[1, 0], 0.0)

def test_preprocessing_pipeline():
    pipeline = PreprocessingPipeline(config_path="config.yaml")
    raw_data = np.array([[1000.0, np.nan], [5000.0, 10000.0]], dtype=float)
    cleaned = pipeline.clean_and_normalize(raw_data)
    
    assert cleaned.shape == (2, 2)
    assert np.isclose(cleaned[0, 0], 0.1)
    assert np.isclose(cleaned[0, 1], 0.0)  # NaN filled with 0
    assert np.isclose(cleaned[1, 0], 0.5)
    assert np.isclose(cleaned[1, 1], 1.0)

def test_feature_extractor_shapes():
    extractor = FeatureExtractor(config_path="config.yaml")
    h, w = 10, 10
    dummy_band = np.random.rand(h, w).astype(np.float32)
    
    t1_bands = {k: dummy_band for k in ["red", "green", "blue", "nir", "swir"]}
    t2_bands = {k: dummy_band for k in ["red", "green", "blue", "nir", "swir"]}
    gt = np.zeros((h, w), dtype=np.int32)
    
    fmaps = extractor.extract_scene_features_2d(t1_bands, t2_bands)
    X, y = extractor.extract_tabular_dataset(fmaps, gt)
    
    assert X.shape == (100, 16)
    assert y.shape == (100,)
    assert not np.isnan(X).any()
