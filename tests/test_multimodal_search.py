import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from src.embeddings import MultimodalEmbeddingEngine, HybridSpectralEmbeddingEngine
from src.vector_store import SatelliteVectorStore
from app.app import app

def test_multimodal_embedding_engine():
    """Verify dual text and vision embedding generation and dimensionality."""
    engine = MultimodalEmbeddingEngine()

    # 1. Text embedding
    v_text = engine.encode_text("large water reservoir near urban settlement")
    assert isinstance(v_text, np.ndarray)
    assert v_text.shape == (512,)
    assert np.isclose(np.linalg.norm(v_text), 1.0, atol=1e-3)

    # 2. Image embedding
    synthetic_patch = np.zeros((128, 128, 3), dtype=np.uint8)
    synthetic_patch[:, :, 2] = 200 # Blue water-like synthetic patch
    v_img = engine.encode_image(synthetic_patch)
    assert isinstance(v_img, np.ndarray)
    assert v_img.shape == (512,)
    assert np.isclose(np.linalg.norm(v_img), 1.0, atol=1e-3)

    # 3. Similarity calculation
    sim = engine.compute_similarity(v_text, v_img)
    assert 0.0 <= sim <= 100.0

def test_satellite_vector_store():
    """Verify vector search, ranking, and spatial filtering."""
    store = SatelliteVectorStore(index_dir="data/vector_index")
    assert store.count() > 0

    # Test Text Search
    results = store.search_by_text("Find large water bodies and reservoirs", top_k=5)
    assert len(results) > 0
    assert results[0]["rank"] == 1
    assert "similarity_pct" in results[0]
    assert "area_m2" in results[0]
    assert "district" in results[0]

    # Test District Spatial Filtering
    chennai_results = store.search_by_text("water and urban areas", district="Chennai", top_k=5)
    for r in chennai_results:
        assert "chennai" in r["district"].lower()

def test_flask_semantic_search_routes():
    """Verify Flask routes for semantic search UI and API."""
    client = app.test_client()

    # 1. GET /semantic-search page
    res_page = client.get("/semantic-search")
    assert res_page.status_code == 200
    assert b"Multimodal Satellite Area Search" in res_page.data

    # 2. GET /api/semantic/status
    res_status = client.get("/api/semantic/status")
    assert res_status.status_code == 200
    status_data = res_status.get_json()
    assert status_data["status"] == "ready"
    assert status_data["total_vectors"] > 0
    assert status_data["embedding_dim"] == 512

    # 3. POST /api/semantic/text-search
    res_api = client.post(
        "/api/semantic/text-search",
        json={"query": "large water bodies and reservoirs", "top_k": 4}
    )
    assert res_api.status_code == 200
    api_data = res_api.get_json()
    assert api_data["status"] == "success"
    assert len(api_data["results"]) == 4
