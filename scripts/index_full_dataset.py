#!/usr/bin/env python3
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference import InferenceEngine
from src.preprocessing.pipeline import OSCDLoader
from src.search_engine import RegionDatabase

def run_large_dataset_indexing():
    print("=" * 80)
    print("🚀 RUNNING LARGE DATASET INFERENCE & DATABASE INDEXING")
    print("=" * 80)
    
    loader = OSCDLoader(config_path="config.yaml")
    images_dir = loader.images_dir
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    all_scenes = sorted([d.name for d in images_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    test_scenes = [s for s in all_scenes if s.startswith("test_")]
    val_scenes = [s for s in all_scenes if s.startswith("val_")]
    
    # Run on all 20 test scenes + 20 validation scenes (40 benchmark scenes)
    target_scenes = test_scenes + val_scenes
    print(f"Found {len(all_scenes)} total dataset scenes.")
    print(f"Processing {len(target_scenes)} benchmark scenes ({len(test_scenes)} Test + {len(val_scenes)} Val)...")
    print("-" * 80)

    engine = InferenceEngine(config_path="config.yaml")
    db = RegionDatabase()

    total_regions_added = 0
    start_all = time.time()

    for idx, scene_id in enumerate(target_scenes, 1):
        try:
            t0 = time.time()
            run_id = f"ds_{scene_id}"
            res_json, regions, run_dir = engine.run_change_detection(
                t1_input=scene_id,
                t2_input=scene_id,
                threshold=0.72,
                min_region_pixels=10,
                run_id=run_id
            )
            elapsed = time.time() - t0
            region_cnt = len(regions)
            total_regions_added += region_cnt
            print(f"[{idx:02d}/{len(target_scenes)}] Scene '{scene_id}': {region_cnt} regions detected ({elapsed:.2f}s)")
        except Exception as e:
            print(f"[{idx:02d}/{len(target_scenes)}] ⚠️ Error processing '{scene_id}': {e}")

    total_elapsed = time.time() - start_all
    print("-" * 80)
    print(f"✅ COMPLETED LARGE DATASET PROCESSING!")
    print(f"   Scenes Processed: {len(target_scenes)}")
    print(f"   Total Regions Extracted & Indexed: {total_regions_added}")
    print(f"   Total Duration: {total_elapsed:.2f} seconds ({total_elapsed/len(target_scenes):.2f}s / scene)")
    print("=" * 80)

if __name__ == "__main__":
    run_large_dataset_indexing()
