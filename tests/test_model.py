import pytest
import numpy as np
from pathlib import Path
from src.models.tuned import XGBoostTuned
from src.inference import InferenceEngine
from src.features.extraction_day3 import FeatureExtractorDay3

@pytest.fixture
def engine():
    return InferenceEngine(config_path="config.yaml")

@pytest.fixture
def model(engine):
    return engine.model

@pytest.fixture
def extractor():
    return FeatureExtractorDay3(config_path="config.yaml")

def test_model_loading_and_features(engine, model):
    """Confirm model loads and checks 32-feature expectation."""
    assert model.model is not None
    assert len(engine.selected_features) == 32

def test_predict_single_sample(model):
    """Predict one 32-feature sample."""
    sample = np.random.rand(1, 32).astype(np.float32)
    prob = model.predict_proba(sample)
    assert prob.shape == (1,)
    assert 0.0 <= prob[0] <= 1.0

def test_predict_scene(engine, model, extractor):
    """Predict probabilities for a full 2D scene grid."""
    h, w = 50, 50
    t1_bands = {k: np.random.rand(h, w).astype(np.float32) for k in ["blue", "green", "red", "nir", "swir"]}
    t2_bands = {k: np.random.rand(h, w).astype(np.float32) for k in ["blue", "green", "red", "nir", "swir"]}
    
    fmaps = extractor.extract_scene_features_2d(t1_bands, t2_bands)
    X = extractor.extract_tabular_dataset(fmaps, engine.selected_features)

    assert X.shape == (h * w, 32)
    # Check finite values constraint: assert np.isfinite(features).all()
    assert np.isfinite(X).all()

    probs = model.predict_proba(X)
    assert probs.shape == (h * w,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = model.predict(X, threshold=0.72)
    assert preds.shape == (h * w,)
    assert set(np.unique(preds)).issubset({0, 1})
