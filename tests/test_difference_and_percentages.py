import pytest
import os
from pathlib import Path
from src.inference import InferenceEngine
from src.reporting import EnvironmentalAuditGenerator
from app.app import app

project_root = Path(__file__).resolve().parent.parent

@pytest.fixture
def engine():
    return InferenceEngine(config_path=str(project_root / "config.yaml"))

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_image_difference_heatmap_and_metrics(engine):
    """Test that image difference detection generates heatmap and quantitative difference metrics."""
    t1_p = project_root / "before_img.png"
    t2_p = project_root / "after_img.png"
    assert t1_p.exists(), "before_img.png must exist in root"
    assert t2_p.exists(), "after_img.png must exist in root"

    res, regions, out_dir = engine.run_change_detection(
        t1_input=str(t1_p),
        t2_input=str(t2_p),
        threshold=0.72,
        run_id="test_diff_auto"
    )

    # 1. Verify diff heatmap was saved
    assert (out_dir / "diff_heatmap.png").exists(), "diff_heatmap.png must be generated"
    assert "diff_heatmap" in res["visualizations"]

    # 2. Verify difference metrics
    diff_metrics = res.get("difference_metrics")
    assert diff_metrics is not None
    assert "mean_abs_diff_percentage" in diff_metrics
    assert "rmse_diff_percentage" in diff_metrics
    assert "pixels_exceeding_5pct_diff" in diff_metrics
    assert diff_metrics["mean_abs_diff_percentage"] >= 0.0

    # 3. Verify top-level percentage values
    assert "change_percentage" in res
    assert "unchanged_percentage" in res
    assert round(res["change_percentage"] + res["unchanged_percentage"], 1) == 100.0
    assert "mean_confidence_percentage" in res

def test_all_regions_and_classes_have_percentages(engine):
    """Verify that every region, semantic class, and transition has valid percentage attributes."""
    t1_p = project_root / "before_img.png"
    t2_p = project_root / "after_img.png"

    res, regions, _ = engine.run_change_detection(
        t1_input=str(t1_p),
        t2_input=str(t2_p),
        threshold=0.72,
        run_id="test_pct_auto"
    )

    # Check semantic breakdown percentages
    for cid, b in res["semantic_breakdown"].items():
        assert "percentage_of_changed_area" in b
        assert "percentage_of_total_scene" in b
        assert 0.0 <= b["percentage_of_changed_area"] <= 100.0
        assert 0.0 <= b["percentage_of_total_scene"] <= 100.0

    # Check region percentages
    for r in regions:
        assert "percentage_of_change_area" in r
        assert "percentage_of_scene_area" in r
        assert "confidence_percentage" in r
        assert 0.0 <= r["percentage_of_change_area"] <= 100.0
        assert 0.0 <= r["percentage_of_scene_area"] <= 100.0
        assert 0.0 <= r["confidence_percentage"] <= 100.0

    # Check transitions percentages
    for tm in res["transition_matrix"]:
        assert "percentage" in tm
        assert 0.0 <= tm["percentage"] <= 100.0

def test_api_difference_check_endpoint(client):
    """Verify /api/difference-check route returns status 200 and complete percentage metrics."""
    response = client.post("/api/difference-check")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert "change_percentage" in data["metrics"]
    assert "unchanged_percentage" in data["metrics"]
    assert "mean_confidence_percentage" in data["metrics"]
    assert "difference_metrics" in data["metrics"]
    assert "diff_heatmap" in data["visualizations"]

def test_demo_sample_pair_route(client):
    """Verify demo route with sample pair redirects to results page."""
    response = client.get("/demo?scene=sample_pair", follow_redirects=True)
    assert response.status_code == 200
    assert b"Semantic Change Detection Results" in response.data
    assert b"Raw Image Difference Heatmap" in response.data

def test_reporting_percentages():
    """Verify government environmental audit report has class and hotspot percentages."""
    gen = EnvironmentalAuditGenerator()
    audit = gen.generate_district_audit("chennai")
    assert audit["status"] == "success"

    for k, v in audit["class_breakdown"].items():
        assert "percentage_of_total_change" in v
        assert "percentage_of_regions" in v
        assert 0.0 <= v["percentage_of_total_change"] <= 100.0

    for h in audit["priority_hotspots"]:
        assert "percentage_of_district_change" in h
        assert "confidence_percentage" in h
