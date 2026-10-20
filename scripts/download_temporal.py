#!/usr/bin/env python3
"""
Temporal Satellite Tile Downloader
Downloads 2020 and 2022 historical imagery from Esri Wayback archive
for all 38 Tamil Nadu districts, matching the existing 2024 tile grid.

Output structure:
  data/tamilnadu_temporal/{district}/year_2020/{TILE_ID}.png
  data/tamilnadu_temporal/{district}/year_2022/{TILE_ID}.png

Each output image is a stitched 1024x1024 (or matching width_px x height_px)
assembled from 256x256 Wayback WMTS tiles.
"""

import os
import io
import math
import json
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Optional

import requests
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DISTRICTS_BASE = PROJECT_ROOT / "dataset_tamilnadu_districts"
OUTPUT_BASE    = PROJECT_ROOT / "data" / "tamilnadu_temporal"

WAYBACK_LAYERS = {
    "2020": "WB_2020_R01",
    "2022": "WB_2022_R01",
}

WMTS_TEMPLATE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/MapServer/WMTS/tile/1.0.0/"
    "{layer}/default/GoogleMapsCompatible/{z}/{y}/{x}"
)

WORKERS      = 8    # parallel threads
RETRY        = 3    # retries per tile
RETRY_DELAY  = 2.0  # seconds between retries
TILE_TIMEOUT = 20   # seconds per HTTP request

# thread-safe progress counter
_lock    = threading.Lock()
_done    = 0
_total   = 0
_skipped = 0
_failed  = 0

# ── Math helpers ───────────────────────────────────────────────────────────────

def lon_lat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y

def tile_bounds(xtile: int, ytile: int, zoom: int) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) for a slippy tile."""
    n = 2.0 ** zoom
    min_lon = xtile / n * 360.0 - 180.0
    max_lon = (xtile + 1) / n * 360.0 - 180.0
    max_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ytile / n))))
    min_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ytile + 1) / n))))
    return min_lon, min_lat, max_lon, max_lat

# ── Session pool ───────────────────────────────────────────────────────────────

_session_local = threading.local()

def get_session() -> requests.Session:
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "SIH-Temporal-Downloader/2.0"})
        _session_local.session = s
    return _session_local.session

# ── Single 256x256 Wayback tile fetch ────────────────────────────────────────

def fetch_wmts_tile(layer: str, z: int, x: int, y: int) -> Optional[Image.Image]:
    url = WMTS_TEMPLATE.format(layer=layer, z=z, x=x, y=y)
    for attempt in range(RETRY):
        try:
            r = get_session().get(url, timeout=TILE_TIMEOUT)
            if r.status_code == 200 and len(r.content) > 500:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
            elif r.status_code == 404:
                return None  # tile doesn't exist in this release
        except Exception:
            pass
        if attempt < RETRY - 1:
            time.sleep(RETRY_DELAY)
    return None

# ── Stitch a region tile ──────────────────────────────────────────────────────

def stitch_region(meta: Dict, layer: str, out_path: Path) -> bool:
    """
    Given tile metadata (bbox + output size), fetch all underlying
    256x256 Wayback tiles and stitch them into one image.
    Returns True on success.
    """
    if out_path.exists():
        return True  # already cached

    zoom      = int(meta.get("zoom_level", 14))
    min_lon   = float(meta["min_lon"])
    min_lat   = float(meta["min_lat"])
    max_lon   = float(meta["max_lon"])
    max_lat   = float(meta["max_lat"])
    out_w     = int(meta.get("width_px",  1024))
    out_h     = int(meta.get("height_px", 1024))

    # Tile range covering this bbox
    x_min, y_min = lon_lat_to_tile(min_lon, max_lat, zoom)  # top-left
    x_max, y_max = lon_lat_to_tile(max_lon, min_lat, zoom)  # bottom-right

    # Build full mosaic of 256px tiles
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    mosaic = Image.new("RGB", (nx * 256, ny * 256), (30, 30, 30))

    any_ok = False
    for ix, tx in enumerate(range(x_min, x_max + 1)):
        for iy, ty in enumerate(range(y_min, y_max + 1)):
            tile_img = fetch_wmts_tile(layer, zoom, tx, ty)
            if tile_img:
                mosaic.paste(tile_img, (ix * 256, iy * 256))
                any_ok = True

    if not any_ok:
        return False

    # Compute crop within the mosaic that matches the exact bbox
    mosaic_min_lon, mosaic_max_lat = tile_bounds(x_min, y_min, zoom)[0], tile_bounds(x_min, y_min, zoom)[3]
    mosaic_max_lon, mosaic_min_lat = tile_bounds(x_max, y_max, zoom)[2], tile_bounds(x_max, y_max, zoom)[1]

    mosaic_w = nx * 256
    mosaic_h = ny * 256
    lon_span = mosaic_max_lon - mosaic_min_lon
    lat_span = mosaic_max_lat - mosaic_min_lat

    crop_x1 = int((min_lon - mosaic_min_lon) / lon_span * mosaic_w)
    crop_y1 = int((mosaic_max_lat - max_lat) / lat_span * mosaic_h)
    crop_x2 = int((max_lon - mosaic_min_lon) / lon_span * mosaic_w)
    crop_y2 = int((mosaic_max_lat - min_lat) / lat_span * mosaic_h)

    crop_x1 = max(0, min(crop_x1, mosaic_w - 1))
    crop_y1 = max(0, min(crop_y1, mosaic_h - 1))
    crop_x2 = max(crop_x1 + 1, min(crop_x2, mosaic_w))
    crop_y2 = max(crop_y1 + 1, min(crop_y2, mosaic_h))

    cropped = mosaic.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    final   = cropped.resize((out_w, out_h), Image.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path, "PNG", optimize=True)
    return True

# ── Worker task ───────────────────────────────────────────────────────────────

def process_tile(meta: Dict, district: str, year: str, layer: str) -> Tuple[str, bool, bool]:
    """Returns (tile_id, was_skipped, success)."""
    global _done, _skipped, _failed

    tile_id  = meta["tile_id"]
    out_dir  = OUTPUT_BASE / district / f"year_{year}"
    out_path = out_dir / f"{tile_id}_{year}.png"

    if out_path.exists():
        with _lock:
            _done += 1
            _skipped += 1
        return tile_id, True, True

    ok = stitch_region(meta, layer, out_path)

    with _lock:
        _done += 1
        if not ok:
            _failed += 1
        if _done % 50 == 0 or _done == _total:
            pct = _done / _total * 100
            print(f"  [{_done:5d}/{_total}] {pct:5.1f}%  skipped={_skipped}  failed={_failed}", flush=True)

    return tile_id, False, ok

# ── Main ──────────────────────────────────────────────────────────────────────

def collect_all_tasks(years: List[str]) -> List[Tuple[Dict, str, str, str]]:
    """Collect all (meta, district, year, layer) tasks."""
    tasks = []
    for dist_dir in sorted(DISTRICTS_BASE.iterdir()):
        if not dist_dir.is_dir():
            continue
        district = dist_dir.name
        tile_dir = dist_dir / "close_aerial_z14"
        if not tile_dir.exists():
            continue
        meta_files = sorted(tile_dir.glob("*_meta.json"))
        for mf in meta_files:
            try:
                with open(mf) as f:
                    meta = json.load(f)
            except Exception:
                continue
            for year in years:
                layer = WAYBACK_LAYERS[year]
                tasks.append((meta, district, year, layer))
    return tasks


def main():
    global _total

    parser = argparse.ArgumentParser(description="Download Wayback temporal tiles")
    parser.add_argument("--years",    default="2020,2022", help="Comma-separated years")
    parser.add_argument("--workers",  type=int, default=WORKERS)
    parser.add_argument("--district", default="all", help="Single district or 'all'")
    args = parser.parse_args()

    years = [y.strip() for y in args.years.split(",") if y.strip() in WAYBACK_LAYERS]
    if not years:
        print("No valid years. Choose from: " + ", ".join(WAYBACK_LAYERS.keys()))
        return

    print(f"\n{'='*60}")
    print(f" Temporal Tile Downloader")
    print(f" Years  : {years}")
    print(f" Workers: {args.workers}")
    print(f" Output : {OUTPUT_BASE}")
    print(f"{'='*60}\n")

    tasks = collect_all_tasks(years)

    if args.district != "all":
        tasks = [(m, d, y, l) for m, d, y, l in tasks if d == args.district]

    _total = len(tasks)
    print(f"Total tasks: {_total}  ({_total // len(years)} tiles × {len(years)} years)\n")

    if _total == 0:
        print("Nothing to download.")
        return

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_tile, meta, dist, year, layer): (meta["tile_id"], year)
            for meta, dist, year, layer in tasks
        }
        for fut in as_completed(futures):
            pass  # progress printed inside worker

    elapsed = time.time() - start
    total_downloaded = _total - _skipped - _failed
    print(f"\n{'='*60}")
    print(f" Done in {elapsed:.0f}s")
    print(f" Downloaded : {total_downloaded}")
    print(f" Skipped    : {_skipped} (already cached)")
    print(f" Failed     : {_failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
