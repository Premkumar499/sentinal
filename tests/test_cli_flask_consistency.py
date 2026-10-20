import pytest
import numpy as np
from pathlib import Path
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

def test_cli_vs_flask_exact_consistency(engine, client):
    """
    Verify CLI inference and Flask API yield 100% identical outputs:
    - change_percentage
    - changed_pixels
    - region_count
    - semantic_breakdown
    - largest_region
    """
    sample_scene = "test_000"

    # 1. Run via CLI InferenceEngine
    cli_json, cli_regions, _ = engine.run_change_detection(
        t1_input=sample_scene,
        t2_input=sample_scene,
        threshold=0.72,
        min_region_pixels=10,
        run_id="consistency_cli"
    )

    # 2. Run via Flask REST API
    api_resp = client.post('/api/detect', json={'sample_scene': sample_scene})
    assert api_resp.status_code == 200
    flask_json = api_resp.get_json()

    # 3. Assert exact equivalence across all metrics
    assert cli_json["changed_pixels"] == flask_json["changed_pixels"]
    assert cli_json["change_percentage"] == flask_json["change_percentage"]
    assert cli_json["region_count"] == flask_json["region_count"]
    assert cli_json["threshold_used"] == flask_json["threshold_used"]
    assert cli_json["largest_region"] == flask_json["largest_region"]

    # Semantic Breakdown equivalence
    for cid in ["1", "2", "3", "4", "5", "6"]:
        cli_class = cli_json["semantic_breakdown"][cid]
        flask_class = flask_json["semantic_breakdown"][cid]
        assert cli_class["pixels"] == flask_class["pixels"]
        assert cli_class["percentage_of_changed_area"] == flask_class["percentage_of_changed_area"]
