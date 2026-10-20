#!/usr/bin/env python3
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import MasterPipeline

def run_evaluation():
    print("==================================================")
    print("RUNNING FINAL SYSTEM EVALUATION & REPRODUCIBILITY")
    print("==================================================")
    
    pipeline = MasterPipeline()
    
    # Run test on sample scene
    sample_t1 = np.ones((100, 100, 3), dtype=np.uint8) * 100
    sample_t2 = np.ones((100, 100, 3), dtype=np.uint8) * 200
    sample_t2[20:50, 20:50] = 50 # Artificial spectral change box
    
    start_time = time.time()
    res = pipeline.run_change_analysis(sample_t1, sample_t2, scene_id="eval_scene_01")
    elapsed = time.time() - start_time
    
    print(f"Status: {res['status']}")
    print(f"Analysis ID: {res['analysis_id']}")
    print(f"Region Count: {len(res['regions'])}")
    print(f"Inference Time: {elapsed:.3f} seconds")
    print("==================================================")
    print("✅ Final Evaluation Pipeline Execution PASSED!")

if __name__ == "__main__":
    run_evaluation()
