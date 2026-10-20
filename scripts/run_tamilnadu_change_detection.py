#!/usr/bin/env python3
"""
Automated 2020 vs 2025 Bi-Temporal Satellite Change Detection Pipeline for Tamil Nadu.

Exclusively pairs real 2020 orthorectified imagery (T1) with real 2025 imagery (T2),
executes the XGBoost semantic change detection engine, extracts connected components,
ranks significance scores, and indexes results into the SQLite search database.

Zero synthetic fallbacks. 100% genuine multi-temporal Earth observation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference import InferenceEngine
from src.search_engine import RegionDatabase
from scripts.download_2020_2025 import stitch_tile


def run_tamilnadu_pipeline(
    districts: Optional[List[str]] = None,
    max_tiles_per_district: int = 50,
    threshold: float = 0.65,
    min_region_pixels: int = 25,
    year1: str = "2020",
    year2: str = "2025",
    auto_download_missing: bool = True
):
    print("=" * 80)
    print("🛰️  TAMIL NADU 2020 vs 2025 BI-TEMPORAL SATELLITE CHANGE DETECTION PIPELINE")
    print("=" * 80)

    base_dir = project_root / "dataset_tamilnadu_districts"
    temporal_base = project_root / "data" / "tamilnadu_temporal"

    if not base_dir.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return

    # Default: ALL 38 Tamil Nadu districts
    if not districts:
        districts = sorted([
            d.name for d in base_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("tamilnadu")
        ])

    print(f"Target Districts:        {len(districts)} districts")
    print(f"Baseline T1:             Year {year1}")
    print(f"Comparison T2:           Year {year2}")
    print(f"Detection Threshold:     {threshold}")
    print(f"Min Region Pixels:       {min_region_pixels}")
    print(f"Max Tiles Per District:  {max_tiles_per_district}")
    print(f"Auto-download Missing:   {auto_download_missing}")
    print("-" * 80)

    engine = InferenceEngine(config_path=str(project_root / "config.yaml"))
    db = RegionDatabase()

    total_tiles_processed = 0
    total_regions_extracted = 0
    total_tiles_skipped = 0
    all_runs_summary = []
    start_total_time = time.time()

    for dist in districts:
        dist_clean = dist.lower().strip()
        close_dir = base_dir / dist_clean / "close_aerial_z14"
        if not close_dir.exists():
            print(f"⚠️ Directory {close_dir} does not exist, skipping.")
            continue

        meta_files = sorted(close_dir.glob("*_meta.json"))
        target_meta_files = meta_files[:max_tiles_per_district]

        print(f"\n📍 District '{dist_clean.upper()}': {len(meta_files)} total grid tiles defined (analyzing up to {len(target_meta_files)}).")

        dist_2020_dir = temporal_base / dist_clean / f"year_{year1}"
        dist_2025_dir = temporal_base / dist_clean / f"year_{year2}"

        for idx, meta_path in enumerate(target_meta_files, 1):
            tile_stem = meta_path.stem.replace("_meta", "")
            
            with open(meta_path) as f:
                meta = json.load(f)

            t1_path = dist_2020_dir / f"{tile_stem}_{year1}.png"
            t1_meta = dist_2020_dir / f"{tile_stem}_{year1}_meta.json"
            t2_path = dist_2025_dir / f"{tile_stem}_{year2}.png"
            t2_meta = dist_2025_dir / f"{tile_stem}_{year2}_meta.json"

            # Auto-download on demand if missing
            if (not t1_path.exists() or not t2_path.exists()) and auto_download_missing:
                print(f"  [{idx}/{len(target_meta_files)}] Fetching {year1} & {year2} tile: {tile_stem}...")
                ok1 = stitch_tile(meta, year1, t1_path, t1_meta)
                ok2 = stitch_tile(meta, year2, t2_path, t2_meta)
                if not (ok1 and ok2):
                    print(f"     ⚠️ Failed to fetch tile pair for {tile_stem}, skipping.")
                    total_tiles_skipped += 1
                    continue
            elif not t1_path.exists() or not t2_path.exists():
                print(f"  [{idx}/{len(target_meta_files)}] Missing imagery for {tile_stem}, skipping.")
                total_tiles_skipped += 1
                continue

            # Verify both images are readable and valid
            im1 = cv2.imread(str(t1_path))
            im2 = cv2.imread(str(t2_path))
            if im1 is None or im2 is None:
                print(f"     ⚠️ Unreadable image for {tile_stem}, skipping.")
                total_tiles_skipped += 1
                continue

            print(f"  [{idx}/{len(target_meta_files)}] Running Change Detection: {tile_stem} ({year1} vs {year2})...")

            run_id = f"tn_{dist_clean}_{tile_stem}_{time.strftime('%M%S')}"
            try:
                t0 = time.time()
                res_json, regions, run_dir = engine.run_change_detection(
                    t1_input=str(t1_path),
                    t2_input=str(t2_path),
                    threshold=threshold,
                    min_region_pixels=min_region_pixels,
                    run_id=run_id,
                    scene_id=tile_stem
                )
                elapsed = time.time() - t0
                region_count = len(regions)
                total_regions_extracted += region_count
                total_tiles_processed += 1

                print(f"     ✅ Changed Pixels: {res_json['changed_pixels']} ({res_json['change_percentage']}%) | Regions: {region_count} | Duration: {elapsed:.2f}s")

                if regions:
                    classes_found = [r.get("Class", "unknown") for r in regions]
                    class_summary = {c: classes_found.count(c) for c in set(classes_found)}
                    print(f"     📊 Detected Classes: {class_summary}")

                all_runs_summary.append({
                    "district": dist_clean,
                    "tile_id": tile_stem,
                    "run_id": run_id,
                    "year1": year1,
                    "year2": year2,
                    "t1_path": str(t1_path),
                    "t2_path": str(t2_path),
                    "changed_pixels": res_json["changed_pixels"],
                    "change_percentage": res_json["change_percentage"],
                    "region_count": region_count,
                    "run_dir": str(run_dir)
                })

            except Exception as e:
                print(f"     ⚠️ Error processing {tile_stem}: {e}")

    total_duration = time.time() - start_total_time
    summary_path = project_root / "outputs" / "tamilnadu_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "year_comparison": f"{year1}_vs_{year2}",
            "total_tiles_processed": total_tiles_processed,
            "total_regions_extracted": total_regions_extracted,
            "total_tiles_skipped": total_tiles_skipped,
            "total_duration_sec": round(total_duration, 2),
            "runs": all_runs_summary
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 TAMIL NADU 2020 vs 2025 BI-TEMPORAL PIPELINE COMPLETE!")
    print(f"   Total Real Tiles Analyzed: {total_tiles_processed}")
    print(f"   Total Semantic Change Regions Detected: {total_regions_extracted}")
    print(f"   Total Execution Time: {total_duration:.2f}s")
    print(f"   Results Summary Saved: {summary_path}")
    print(f"   Database Updated: {project_root / 'data' / 'change_regions.db'}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tamil Nadu 2020 vs 2025 Bi-Temporal Change Detection.")
    parser.add_argument("--districts", nargs="+", default=None, help="Districts to analyze (default: all 38)")
    parser.add_argument("--max-tiles", type=int, default=50, help="Max tiles per district to process")
    parser.add_argument("--threshold", type=float, default=0.65, help="Change detection probability threshold")
    parser.add_argument("--min-pixels", type=int, default=25, help="Min connected pixels for a change region")
    parser.add_argument("--year1", type=str, default="2020", help="T1 baseline year (default: 2020)")
    parser.add_argument("--year2", type=str, default="2025", help="T2 comparison year (default: 2025)")
    parser.add_argument("--no-download", action="store_true", help="Do not auto-download missing tiles")
    args = parser.parse_args()

    run_tamilnadu_pipeline(
        districts=args.districts,
        max_tiles_per_district=args.max_tiles,
        threshold=args.threshold,
        min_region_pixels=args.min_pixels,
        year1=args.year1,
        year2=args.year2,
        auto_download_missing=not args.no_download
    )
