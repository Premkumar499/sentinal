#!/usr/bin/env python3
"""
Batch fix for master catalog CSV, master catalog JSON, and all individual tile meta JSON files.
Converts legacy Windows file paths into valid local Linux paths.
"""

import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_BASE = PROJECT_ROOT / "dataset_tamilnadu_districts"


def fix_master_csv():
    csv_file = DATASET_BASE / "tamilnadu_close_aerial_master_catalog.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found.")
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        return

    fieldnames = list(reader[0].keys())
    updated_rows = 0

    for row in reader:
        dk = row.get("district_key", "")
        tid = row.get("tile_id", "")
        target_path = str(DATASET_BASE / dk / "close_aerial_z14" / f"{tid}.png")
        if row.get("file_path") != target_path:
            row["file_path"] = target_path
            updated_rows += 1

    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reader)

    print(f"✅ Fixed master CSV: updated {updated_rows}/{len(reader)} rows -> {csv_file}")


def fix_master_json():
    json_file = DATASET_BASE / "tamilnadu_close_aerial_master_catalog.json"
    if not json_file.exists():
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated = 0
    items = data if isinstance(data, list) else data.get("tiles", data.get("features", []))

    for item in items:
        # Check dict structure
        props = item.get("properties", item)
        dk = props.get("district_key", "")
        tid = props.get("tile_id", "")
        if dk and tid:
            target_path = str(DATASET_BASE / dk / "close_aerial_z14" / f"{tid}.png")
            if props.get("file_path") != target_path:
                props["file_path"] = target_path
                updated += 1

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Fixed master JSON: updated {updated} entries -> {json_file}")


def fix_individual_meta_jsons():
    count = 0
    fixed = 0
    for meta_file in DATASET_BASE.rglob("*_meta.json"):
        if "year_" in str(meta_file):
            continue
        count += 1
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            dk = d.get("district_key", "")
            tid = d.get("tile_id", "")
            if dk and tid:
                target_path = str(DATASET_BASE / dk / "close_aerial_z14" / f"{tid}.png")
                if d.get("file_path") != target_path:
                    d["file_path"] = target_path
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(d, f, indent=2)
                    fixed += 1
        except Exception as e:
            print(f"Error reading {meta_file}: {e}")

    print(f"✅ Scanned {count} individual tile meta files, fixed {fixed} paths.")


if __name__ == "__main__":
    print("Fixing dataset paths...")
    fix_master_csv()
    fix_master_json()
    fix_individual_meta_jsons()
    print("All paths normalized to Linux filesystem.")
