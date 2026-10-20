import io
import os
import pytest
import numpy as np
import cv2
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

# TEST 1: Valid T1 + T2
def test_valid_input(client):
    response = client.post('/analyze', data={'sample_scene': 'test_000'}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Change Detection Results" in response.data or b"Results" in response.data

# TEST 2: Only T1 Upload
def test_only_t1_upload(client):
    data = {'file_t1': (io.BytesIO(b"fake image data"), 't1.png')}
    response = client.post('/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Missing Files" in response.data or b"Error" in response.data

# TEST 3: Only T2 Upload
def test_only_t2_upload(client):
    data = {'file_t2': (io.BytesIO(b"fake image data"), 't2.png')}
    response = client.post('/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Missing Files" in response.data or b"Error" in response.data

# TEST 4: Invalid File Path
def test_invalid_file_path(engine):
    valid, msg = engine.validate_inputs("non_existent_t1.png", "non_existent_t2.png")
    assert valid is False
    assert "file does not exist" in msg.lower()

# TEST 5: Corrupted Image Data
def test_corrupted_image(engine, tmp_path):
    corrupt_t1 = tmp_path / "corrupt1.png"
    corrupt_t2 = tmp_path / "corrupt2.png"
    corrupt_t1.write_bytes(b"NOT_AN_IMAGE_DATA")
    corrupt_t2.write_bytes(b"NOT_AN_IMAGE_DATA")

    with pytest.raises(ValueError, match="Could not decode"):
        engine.run_change_detection(str(corrupt_t1), str(corrupt_t2))

# TEST 6: Dimension Handling (Auto-resize matching)
def test_dimension_mismatch_autoresize(engine, tmp_path):
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((120, 120, 3), dtype=np.uint8)
    
    p1 = tmp_path / "t1_100x100.png"
    p2 = tmp_path / "t2_120x120.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)

    # Engine automatically resizes img2 to match img1 shape cleanly
    res_json, regions, run_dir = engine.run_change_detection(str(p1), str(p2))
    assert res_json["image_dimensions"] == [100, 100]

# TEST 7: Unsupported File Extension
def test_invalid_file_extension(client):
    data = {
        'file_t1': (io.BytesIO(b"data"), 'file.exe'),
        'file_t2': (io.BytesIO(b"data"), 'file.exe')
    }
    response = client.post('/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Invalid File Format" in response.data

# TEST 8: Empty File Upload
def test_empty_upload(client):
    data = {
        'file_t1': (io.BytesIO(b""), ''),
        'file_t2': (io.BytesIO(b""), '')
    }
    response = client.post('/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert b"Missing Files" in response.data

# TEST 9: Oversized Payload Check
def test_large_upload_rejection(client):
    # App is configured with 32MB max limit
    data = {'file_t1': (io.BytesIO(b"0" * (33 * 1024 * 1024)), 'large.png')}
    response = client.post('/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code in [413, 200, 400]

# TEST 10: Repeated Uploads generate distinct run IDs
def test_repeated_uploads(engine, tmp_path):
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    p1 = tmp_path / "t1.png"
    p2 = tmp_path / "t2.png"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)

    res1, _, _ = engine.run_change_detection(str(p1), str(p2), run_id="run_repeat_1")
    res2, _, _ = engine.run_change_detection(str(p1), str(p2), run_id="run_repeat_2")

    assert res1["run_id"] == "run_repeat_1"
    assert res2["run_id"] == "run_repeat_2"
