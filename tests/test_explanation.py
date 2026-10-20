import pytest
from pathlib import Path
from src.explanation import ExplanationEngine, get_global_feature_importance, load_day13_config
from app.app import app

def test_config_freeze():
    cfg = load_day13_config()
    assert cfg["model_settings"]["threshold"] == 0.72
    assert cfg["model_settings"]["model_version"] == "xgboost_v1.0_frozen"

def test_global_feature_importance():
    features = get_global_feature_importance()
    assert len(features) >= 4
    assert features[0]["feature"] == "delta_NIR"
    assert "importance" in features[0]
    assert Path("outputs/explanations/global_feature_importance.json").exists()

def test_explanation_engine_generate():
    engine = ExplanationEngine()
    sample_region = {
        "Region ID": 1,
        "Class": "road",
        "Change Type": "expansion",
        "Area": 500,
        "Significance Score": 0.87,
        "NDVI Difference": 0.18,
        "NDBI Difference": 0.14
    }
    exp = engine.generate_region_explanation(sample_region, scene_id="scene_test_01")

    assert exp["region_id"] == 1
    assert exp["semantic_class"] == "road"
    assert exp["change_type"] == "expansion"
    assert exp["confidence_category"] == "HIGH_CONFIDENCE_CHANGE"
    assert exp["quality_status"] == "PASS"
    assert "delta_NDVI" in exp["feature_evidence"]
    assert "what_not_to_conclude" in exp["change_story"]

def test_explanation_consistency():
    """Verify that generating an explanation 3 times produces identical outputs."""
    engine = ExplanationEngine()
    sample_region = {"Region ID": 5, "Class": "building", "Change Type": "new", "Area": 250, "Significance Score": 0.82}
    
    exp1 = engine.generate_region_explanation(sample_region, scene_id="scene_consistency")
    exp2 = engine.generate_region_explanation(sample_region, scene_id="scene_consistency")
    exp3 = engine.generate_region_explanation(sample_region, scene_id="scene_consistency")

    assert exp1 == exp2 == exp3

def test_flask_explanation_api_endpoints():
    client = app.test_client()

    # 1. Model Info Page
    res_mod = client.get("/model")
    assert res_mod.status_code == 200
    assert b"Model Audit & Global Feature Interpretability" in res_mod.data

    # 2. GET Global Features API
    res_feat = client.get("/api/model/features")
    assert res_feat.status_code == 200
    assert res_feat.json["status"] == "success"
    assert len(res_feat.json["features"]) >= 4

    # 3. GET Region Explanation API
    res_exp = client.get("/api/regions/scene_01/1/explanation")
    assert res_exp.status_code == 200
    assert res_exp.json["status"] == "success"
    assert "explanation" in res_exp.json

    # 4. GET Export Explanation Text API
    res_exp_txt = client.get("/api/regions/scene_01/1/export_explanation?format=text")
    assert res_exp_txt.status_code == 200
    assert b"REGION #1 EXPLAINABILITY" in res_exp_txt.data
