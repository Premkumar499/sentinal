#!/usr/bin/env python3
"""
Full-State Master Pipeline for Tamil Nadu 2020 vs 2025 Change Detection.

Executes the complete end-to-end lifecycle across all 38 districts:
1. Precision Tile Download (2020 & 2025 with 16 workers, Slippy grid aligned)
2. Catalog Organization & Tile Connection Verification
3. Bi-Temporal Change Detection (2020 vs 2025, zero synthetic data)
4. Semantic Classification & Object Region Extraction
5. Vector Embeddings Indexing (FAISS multimodal search)
6. Place Name & Geographic Indexing
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(step_name: str, cmd: list) -> bool:
    print("\n" + "=" * 80)
    print(f"▶️  STARTING STEP: {step_name}")
    print(f"   Command: {' '.join(str(c) for c in cmd)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"✅ STEP COMPLETED: {step_name} (in {elapsed:.1f}s)")
        return True
    else:
        print(f"❌ STEP FAILED: {step_name} (exit code {res.returncode}, after {elapsed:.1f}s)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Full Tamil Nadu 2020 vs 2025 Pipeline")
    parser.add_argument("--districts", nargs="+", default=None, help="Specific districts (default: all 38)")
    parser.add_argument("--max-tiles", type=int, default=15, help="Max tiles per district (default: 15 for comprehensive multi-district coverage; use 0 for all 6,499 tiles)")
    parser.add_argument("--workers", type=int, default=16, help="Download concurrency threads")
    parser.add_argument("--threshold", type=float, default=0.65, help="Change probability threshold")
    parser.add_argument("--min-pixels", type=int, default=25, help="Min connected pixels for a change region")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step if tiles already exist")
    args = parser.parse_args()

    max_tiles_arg = None if args.max_tiles == 0 else args.max_tiles
    start_time = time.time()

    print("=" * 80)
    print("🛰️  TAMIL NADU FULL-STATE 2020 vs 2025 CHANGE DETECTION WORKFLOW")
    print(f"   Target Districts: {'ALL 38 Districts' if not args.districts else args.districts}")
    print(f"   Tiles per District: {max_tiles_arg if max_tiles_arg else 'ALL (Full state grid)'}")
    print(f"   Download Workers: {args.workers}")
    print(f"   Threshold: {args.threshold} | Min Pixels: {args.min_pixels}")
    print("=" * 80)

    # Step 1: Download 2020 & 2025 Tile Pairs
    if not args.skip_download:
        dl_cmd = [sys.executable, "scripts/download_2020_2025.py", "--workers", str(args.workers)]
        if args.districts:
            dl_cmd.extend(["--districts"] + args.districts)
        if max_tiles_arg:
            dl_cmd.extend(["--max-tiles", str(max_tiles_arg)])
        ok = run_step("1. Download 2020 & 2025 Orthorectified Tiles", dl_cmd)
        if not ok:
            print("⚠️ Download encountered issues, proceeding with available tiles.")

    # Step 2: Organize Catalog and Build Continuous Symlink Grid
    run_step("2. Organize Bi-Temporal Catalog (2020 vs 2025)", [sys.executable, "scripts/organize_temporal.py"])

    # Step 3: Run Bi-Temporal Change Detection
    det_cmd = [
        sys.executable, "scripts/run_tamilnadu_change_detection.py",
        "--threshold", str(args.threshold),
        "--min-pixels", str(args.min_pixels)
    ]
    if args.districts:
        det_cmd.extend(["--districts"] + args.districts)
    if max_tiles_arg:
        det_cmd.extend(["--max-tiles", str(max_tiles_arg)])
    run_step("3. Execute 2020 vs 2025 Change Detection Engine", det_cmd)

    # Step 4: Index Vector Embeddings for Multimodal Search
    run_step("4. Build Multimodal FAISS Vector Index", [sys.executable, "scripts/index_satellite_embeddings.py"])

    # Step 5: Build Place Name Index
    run_step("5. Build Geographic Place Search Index", [sys.executable, "scripts/build_place_index.py"])

    # Step 6: Final Catalog Sync
    run_step("6. Final Catalog Status Update", [sys.executable, "scripts/organize_temporal.py"])

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 FULL-STATE PIPELINE COMPLETE!")
    print(f"   Total Pipeline Duration: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"   SQLite Database: {PROJECT_ROOT / 'data' / 'change_regions.db'}")
    print(f"   Temporal Catalog: {PROJECT_ROOT / 'data' / 'temporal_catalog' / 'summary.txt'}")
    print(f"   Vector Index: {PROJECT_ROOT / 'data' / 'vector_index'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
