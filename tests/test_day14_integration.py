import pytest
import numpy as np
from app.app import app
from src.pipeline import MasterPipeline, AnalysisError

def test_health_check_endpoints():
    client = app.test_client()

    # /health endpoint
    res_h = client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json["status"] == "ok"

    # /api/system/status endpoint
    res_sys = client.get("/api/system/status")
    assert res_sys.status_code == 200
    assert res_sys.json["model"] == "ready"
    assert res_sys.json["semantic_layer"] == "ready"

def test_custom_error_pages():
    client = app.test_client()

    # 404 handler
    res_404 = client.get("/nonexistent_page_12345")
    assert res_404.status_code == 404
    assert b"Page Not Found" in res_404.data

def test_input_validation_error_codes():
    pipeline = MasterPipeline()

    # Invalid input (None)
    with pytest.raises(AnalysisError) as exc_info:
        pipeline.validate_input(None, None)
    assert exc_info.value.code == "INVALID_INPUT"

    # Invalid input (NaNs)
    bad_img = np.ones((50, 50, 3))
    bad_img[10, 10, 0] = np.nan
    with pytest.raises(AnalysisError) as exc_info2:
        pipeline.validate_input(bad_img, bad_img)
    assert exc_info2.value.code == "INVALID_INPUT"
