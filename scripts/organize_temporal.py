#!/usr/bin/env python3
"""
Organizes downloaded temporal tiles into a clean year-wise + district-wise
catalog structure and generates summary reports for strictly 2020 and 2025.

Structure:
  data/tamilnadu_temporal/{district}/year_2020/{TILE_ID}_2020.png
  data/tamilnadu_temporal/{district}/year_2025/{TILE_ID}_2025.png

Output catalog:
  data/temporal_catalog/
    catalog.json          ← master index (district → year → tiles)
    summary.txt           ← human-readable summary table
    by_year/
      2020/
        chennai/          ← symlinks
      2025/
        chennai/
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TEMPORAL_BASE  = PROJECT_ROOT / "data" / "tamilnadu_temporal"
DISTRICTS_BASE = PROJECT_ROOT / "dataset_tamilnadu_districts"
CATALOG_DIR    = PROJECT_ROOT / "data" / "temporal_catalog"
CATALOG_FILE   = CATALOG_DIR / "catalog.json"
SUMMARY_FILE   = CATALOG_DIR / "summary.txt"
BY_YEAR_DIR    = CATALOG_DIR / "by_year"

YEARS = ["2020", "2025"]


def collect_catalog() -> dict:
    """
    Walk all districts and years (2020 & 2025), collect tile paths into a master catalog.
    """
    catalog = defaultdict(lambda: defaultdict(dict))

    # All 38 districts defined in dataset_tamilnadu_districts
    all_districts = []
    if DISTRICTS_BASE.exists():
        all_districts = sorted([d.name for d in DISTRICTS_BASE.iterdir() if d.is_dir() and not d.name.startswith(".")])

    for dist in all_districts:
        for year in YEARS:
            year_dir = TEMPORAL_BASE / dist / f"year_{year}"
            if year_dir.exists():
                for png in sorted(year_dir.glob("*.png")):
                    tile_id = png.stem.replace(f"_{year}", "")
                    catalog[dist][year][tile_id] = str(png)
            else:
                catalog[dist][year] = {}

    return {d: dict(y) for d, y in catalog.items()}


def build_by_year_symlinks(catalog: dict):
    """
    Create by_year/{year}/{district}/ with symlinks pointing to real files.
    """
    for district, years in catalog.items():
        for year, tiles in years.items():
            if not tiles:
                continue
            link_dir = BY_YEAR_DIR / year / district
            link_dir.mkdir(parents=True, exist_ok=True)
            for tile_id, real_path in tiles.items():
                real = Path(real_path)
                link = link_dir / real.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                try:
                    link.symlink_to(real)
                except Exception:
                    pass


def generate_summary(catalog: dict) -> str:
    all_districts = sorted(catalog.keys())
    lines = []
    lines.append("=" * 64)
    lines.append(" Tamil Nadu Bi-Temporal Satellite Dataset — 2020 vs 2025 Catalog")
    lines.append("=" * 64)
    lines.append(f"\n{'District':<24} {'2020':>8} {'2025':>8}  Status")
    lines.append("-" * 64)

    total = defaultdict(int)
    complete = 0
    partial  = 0
    missing  = 0

    for dist in all_districts:
        years = catalog.get(dist, {})
        c2020 = len(years.get("2020", {}))
        c2025 = len(years.get("2025", {}))
        total["2020"] += c2020
        total["2025"] += c2025

        has_both = c2020 > 0 and c2025 > 0 and c2020 == c2025
        has_any = c2020 > 0 or c2025 > 0

        if has_both:
            status = "✅ Complete"
            complete += 1
        elif has_any:
            status = "⏳ Partial"
            partial += 1
        else:
            status = "❌ Pending"
            missing += 1

        lines.append(f"  {dist:<22} {c2020:>8} {c2025:>8}  {status}")

    lines.append("-" * 64)
    lines.append(f"  {'TOTAL':<22} {total['2020']:>8} {total['2025']:>8}")
    lines.append("")
    lines.append(f"  Districts complete (2020=2025) : {complete}/{len(all_districts)}")
    lines.append(f"  Districts partial              : {partial}/{len(all_districts)}")
    lines.append(f"  Districts pending download     : {missing}/{len(all_districts)}")
    lines.append("")
    lines.append(f"  Grand total tiles              : {sum(total.values()):,}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    print("\n🗂️  Organizing 2020 vs 2025 bi-temporal dataset...\n")

    catalog = collect_catalog()
    print(f"  Found {len(catalog)} districts tracked")

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"  Saved catalog → {CATALOG_FILE}")

    build_by_year_symlinks(catalog)
    print(f"  Built by_year symlink tree → {BY_YEAR_DIR}")

    summary = generate_summary(catalog)
    with open(SUMMARY_FILE, "w") as f:
        f.write(summary)
    print(f"  Saved summary → {SUMMARY_FILE}\n")
    print(summary)


if __name__ == "__main__":
    main()
