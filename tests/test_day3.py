import numpy as np
import pytest
from src.models.tuned import XGBoostTuned
from src.features.extraction_day3 import FeatureExtractorDay3
from run_day3 import post_process_mask
from src.models.semantic import get_placeholder_semantic_mask, compute_semantic_change, compute_road_change

def test_xgb_tuned_training_and_prediction():
    # Setup simple toy classification problem
    np.random.seed(42)
    X_train = np.random.rand(100, 5).astype(np.float32)
    y_train = np.random.choice([0, 1], size=100, p=[0.9, 0.1]).astype(np.int32)
    
    model = XGBoostTuned(config_path="config.yaml", scale_pos_weight=9.0)
    model.train(X_train, y_train)
    
    assert model.is_trained
    
    # Test predictions
    X_test = np.random.rand(10, 5).astype(np.float32)
    probs = model.predict_proba(X_test)
    assert len(probs) == 10
    assert np.all(probs >= 0) and np.all(probs <= 1)
    
    # Predict with default/custom thresholds
    preds_def = model.predict(X_test, threshold=0.5)
    assert len(preds_def) == 10
    assert set(preds_def).issubset({0, 1})
    
    preds_low = model.predict(X_test, threshold=0.1)
    assert len(preds_low) == 10

def test_xgb_tuned_feature_importance():
    np.random.seed(42)
    X_train = np.random.rand(50, 4).astype(np.float32)
    y_train = np.random.choice([0, 1], size=50).astype(np.int32)
    
    model = XGBoostTuned(config_path="config.yaml")
    model.train(X_train, y_train)
    
    names = ["feat_A", "feat_B", "feat_C", "feat_D"]
    importances = model.get_feature_importance(feature_names=names)
    
    assert len(importances) == 4
    for name in names:
        assert name in importances
        assert isinstance(importances[name], (float, np.float32, np.float64))

def test_feature_extractor_day3():
    extractor = FeatureExtractorDay3(config_path="config.yaml")
    shape = (100, 100)
    
    # Mock band data
    t1_bands = {k: np.random.rand(*shape).astype(np.float32) for k in ["red", "green", "blue", "nir", "swir"]}
    t2_bands = {k: np.random.rand(*shape).astype(np.float32) for k in ["red", "green", "blue", "nir", "swir"]}
    
    fmaps = extractor.extract_scene_features_2d(t1_norm=t1_bands, t2_norm=t2_bands) if hasattr(extractor, 'extract_scene_features_2d_norm') else extractor.extract_scene_features_2d(t1_bands, t2_bands)
    
    # Verify presence of base, temporal, and spatial features
    assert "R_T1" in fmaps
    assert "R_abs_diff" in fmaps
    assert "R_rel_change" in fmaps
    assert "mean_3x3_T1_red" in fmaps
    assert "std_3x3_RGB_diff_mag" in fmaps
    
    # Test tabular generation
    cols = ["R_T1", "R_abs_diff", "mean_3x3_T1_red"]
    gt = np.random.choice([0, 1], size=shape).astype(np.int32)
    X, y = extractor.extract_tabular_dataset(fmaps, cols, gt)
    
    assert X.shape == (10000, 3)
    assert y.shape == (10000,)

def test_post_processing():
    np.random.seed(42)
    # Binary mask with random noise and a block of change
    mask = np.zeros((30, 30), dtype=np.int32)
    mask[10:20, 10:20] = 1 # True change block
    mask[2, 2] = 1 # Small isolated positive noise (should be removed by opening/min_size)
    mask[15, 15] = 0 # Small hole (should be filled by closing)
    
    processed = post_process_mask(mask, open_kernel=3, close_kernel=3, min_size=5)
    
    # Verify noise removal
    assert processed[2, 2] == 0
    # Verify hole filling
    assert processed[15, 15] == 1
    # Verify change block is preserved
    assert processed[11, 11] == 1

def test_semantic_interfaces():
    shape = (50, 50)
    change_mask = np.zeros(shape, dtype=np.int32)
    change_mask[20:30, 20:30] = 1
    
    road_mask = get_placeholder_semantic_mask(shape, "road")
    building_mask = get_placeholder_semantic_mask(shape, "building")
    
    # Verify shape
    assert road_mask.shape == shape
    assert building_mask.shape == shape
    
    # Verify compute semantic change
    road_change = compute_road_change(change_mask, road_mask)
    assert road_change.shape == shape
    assert np.all(road_change <= change_mask)
