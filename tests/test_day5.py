import os
import sys
import json
import pytest
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference import InferenceEngine
from app.app import app

@pytest.fixture
def engine():
    return InferenceEngine(config_path="config.yaml")

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_inference_engine_sample_scene(engine):
    """Test full inference pipeline on sample scene test_000."""
    result_json, regions, run_dir = engine.run_change_detection(
        t1_input="test_000",
        t2_input="test_000",
        threshold=0.72,
        min_region_pixels=10,
        run_id="test_run_000"
    )

    assert result_json["status"] == "success"
    assert result_json["scene_id"] == "test_000"
    assert result_json["threshold_used"] == 0.72
    assert "change_detected" in result_json
    assert "semantic_breakdown" in result_json
    assert len(regions) > 0

    # Verify visual outputs exist
    for img_name in ["before.png", "after.png", "change_map.png", "semantic_map.png", "semantic_change_map.png", "overlay.png", "regions.csv", "result.json", "result_test_run_000.zip"]:
        assert (run_dir / img_name).exists(), f"Missing output file: {img_name}"

def test_flask_homepage(client):
    """Test Flask GET / route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Semantic Satellite Change Detection" in response.data

def test_flask_analyze_sample(client):
    """Test Flask POST /analyze route with sample scene."""
    response = client.post('/analyze', data={'sample_scene': 'test_000'}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Analysis Results" in response.data or b"Change Detection Results" in response.data

def test_flask_api_detect(client):
    """Test REST API endpoint POST /api/detect."""
    response = client.post('/api/detect', json={'sample_scene': 'test_000'})
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["scene_id"] == "test_000"
    assert "changed_pixels" in data

def test_cli_vs_flask_consistency(engine, client):
    """Verify CLI and Flask API outputs are numerically equivalent on test_000."""
    # CLI Run
    cli_json, _, _ = engine.run_change_detection(t1_input="test_000", t2_input="test_000", run_id="cli_check")

    # API Run
    api_resp = client.post('/api/detect', json={'sample_scene': 'test_000'})
    api_json = api_resp.get_json()

    assert cli_json["changed_pixels"] == api_json["changed_pixels"]
    assert cli_json["change_percentage"] == api_json["change_percentage"]
    assert cli_json["threshold_used"] == api_json["threshold_used"]

def test_input_validation_missing_file(engine):
    """Test graceful validation on non-existent files."""
    valid, msg = engine.validate_inputs("non_existent_1.png", "non_existent_2.png")
    assert valid is False
    assert "does not exist" in msg
