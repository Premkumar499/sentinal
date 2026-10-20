import pytest
import numpy as np
from pathlib import Path
from src.pipeline import MasterPipeline

def test_master_pipeline_end_to_end(tmp_path):
    pipeline = MasterPipeline()

    # Synthetic T1 and T2 images with artificial change area
    t1 = np.ones((120, 120, 3), dtype=np.uint8) * 100
    t2 = np.ones((120, 120, 3), dtype=np.uint8) * 100
    t2[30:70, 30:70] = [200, 50, 50] # Artificial red-shift spectral change

    res = pipeline.run_change_analysis(t1, t2, scene_id="scene_e2e_test")

    assert res["status"] == "success"
    assert "analysis_id" in res
    assert "regions" in res
    assert "statistics" in res
    
    run_dir = Path(res["run_dir"])
    assert run_dir.exists()
    assert (run_dir / "temp_t1.png").exists()
    assert (run_dir / "temp_t2.png").exists()
    assert (run_dir / "session_manifest.json").exists()
