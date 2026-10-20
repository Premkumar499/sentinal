"""
Temporal Satellite / Aerial Ingestion Engine for Tamil Nadu Datasets.
Fetches historical high-resolution orthorectified imagery from the open
World Imagery Wayback archive corresponding to any Tamil Nadu Zoom 14 tile.
Zero API keys or credentials required.
"""

import os
import io
import math
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image

# Stable releases in Wayback archive (release numbers)
WAYBACK_RELEASES = {
    "2020": 29260,  # 2020-12-16
    "2022": 19570,  # 2022
    "2024": 46271,  # 2024
    "2025": 13192,  # 2025-12-18
}

MAPSERVER_URL_TEMPLATE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
    "WMTS/1.0.0/default028mm/MapServer/tile/{release_num}/{zoom}/{y}/{x}"
)


def lon_lat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Convert longitude and latitude to Slippy Map tile (X, Y) at given zoom."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_to_lon_lat(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Convert Slippy Map tile (X, Y) to (latitude, longitude) of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


class TemporalFetcher:
    """Automated fetcher for multi-temporal historical satellite/aerial tiles."""

    def __init__(self, cache_dir: Optional[Path] = None, default_year: str = "2020"):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/tamilnadu_temporal")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_year = default_year
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (SIH-Satellite-Change-Detection-Pipeline/2.0)"
        })

    def get_release_for_year(self, year: str) -> int:
        """Resolve year string to Wayback release number."""
        year_str = str(year).strip()
        if year_str in WAYBACK_RELEASES:
            return WAYBACK_RELEASES[year_str]
        try:
            return int(year_str)
        except ValueError:
            return WAYBACK_RELEASES.get(self.default_year, 29260)

    def is_online(self, timeout: float = 1.5) -> bool:
        """Fast check to verify if the remote imagery server is reachable."""
        if hasattr(self, "_online_status"):
            return self._online_status
        try:
            import socket
            sock = socket.socket()
            sock.settimeout(timeout)
            sock.connect(("wayback.maptiles.arcgis.com", 443))
            sock.close()
            self._online_status = True
        except Exception:
            self._online_status = False
        return self._online_status

    def fetch_tile_xyz(self, layer: Any, zoom: int, x: int, y: int, max_retries: int = 2) -> Optional[Image.Image]:
        """Fetch a single 256x256 image tile from Wayback MapServer."""
        if not self.is_online():
            return None
        rel_num = layer if isinstance(layer, int) else self.get_release_for_year(str(layer))
        url = MAPSERVER_URL_TEMPLATE.format(release_num=rel_num, zoom=zoom, x=x, y=y)
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, timeout=5)
                if resp.status_code == 200 and len(resp.content) > 400:
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                time.sleep(0.2)
        return None

    def fetch_temporal_tile(
        self,
        meta_dict_or_path: Any,
        year: str = "2018",
        output_dir: Optional[Path] = None,
        force_redownload: bool = False
    ) -> Tuple[Path, Path]:
        """
        Fetch and stitch historical counterpart for a given Tamil Nadu Zoom 14 tile.
        
        Args:
            meta_dict_or_path: Either a dict containing tile metadata or path to *_meta.json
            year: Historical year (e.g. '2014', '2016', '2018', '2020')
            output_dir: Where to save stitched image and metadata
            force_redownload: If True, overwrite existing cached file
            
        Returns:
            (saved_image_path, saved_meta_path)
        """
        # Load metadata
        if isinstance(meta_dict_or_path, (str, Path)):
            meta_path = Path(meta_dict_or_path)
            with open(meta_path, "r") as f:
                meta = json.load(f)
        else:
            meta = dict(meta_dict_or_path)

        tile_id = meta.get("tile_id", "UNKNOWN_TILE")
        district_key = meta.get("district_key", meta.get("district", "tamilnadu")).lower().replace(" ", "_")
        bbox = meta.get("bbox", [])
        zoom = meta.get("zoom_level", 14)

        if not bbox or len(bbox) < 4:
            raise ValueError(f"Invalid bbox in metadata for tile {tile_id}: {bbox}")

        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]

        if not self.is_online():
            raise ConnectionError("Remote imagery server is currently unreachable.")

        # Calculate Slippy Map tile grid coordinates (3x3 grid for 768x768 px)
        # Top-left tile
        x_start, y_start = lon_lat_to_tile(min_lon, max_lat, zoom)
        
        # Verify 3x3 grid aligns with width_px and height_px
        expected_w = meta.get("width_px", 768)
        expected_h = meta.get("height_px", 768)
        grid_cols = expected_w // 256
        grid_rows = expected_h // 256

        # Determine output location
        if output_dir is None:
            output_dir = self.cache_dir / district_key / f"temporal_{year}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        target_img_name = f"{tile_id}_T1_{year}.png"
        target_meta_name = f"{tile_id}_T1_{year}_meta.json"
        img_out_path = output_dir / target_img_name
        meta_out_path = output_dir / target_meta_name

        if img_out_path.exists() and meta_out_path.exists() and not force_redownload:
            return img_out_path, meta_out_path

        release_num = self.get_release_for_year(year)
        stitched = Image.new("RGB", (expected_w, expected_h))

        for r_idx in range(grid_rows):
            cur_y = y_start + r_idx
            for c_idx in range(grid_cols):
                cur_x = x_start + c_idx
                tile_im = self.fetch_tile_xyz(layer=release_num, zoom=zoom, x=cur_x, y=cur_y)
                if tile_im is None:
                    # Fallback to dark neutral if unavailable
                    tile_im = Image.new("RGB", (256, 256), color=(20, 20, 20))
                stitched.paste(tile_im, (c_idx * 256, r_idx * 256))

        # Save historical stitched image
        stitched.save(img_out_path, "PNG")

        # Create companion metadata
        hist_meta = {
            "tile_id": f"{tile_id}_T1_{year}",
            "parent_tile_id": tile_id,
            "district": meta.get("district", ""),
            "district_key": district_key,
            "temporal_role": "T1_historical",
            "historical_year": year,
            "wayback_layer": layer,
            "zoom_level": zoom,
            "ground_resolution_m_per_px": meta.get("ground_resolution_m_per_px", 9.3),
            "width_px": expected_w,
            "height_px": expected_h,
            "bbox": bbox,
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
            "center_lat": meta.get("center_lat", (min_lat + max_lat) / 2.0),
            "center_lon": meta.get("center_lon", (min_lon + max_lon) / 2.0),
            "area_sq_km": meta.get("area_sq_km", 51.0),
            "crs": meta.get("crs", "EPSG:4326 (WGS84) / EPSG:3857"),
            "source": "ESRI World Imagery Wayback",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        with open(meta_out_path, "w") as f:
            json.dump(hist_meta, f, indent=2)

        return img_out_path, meta_out_path
