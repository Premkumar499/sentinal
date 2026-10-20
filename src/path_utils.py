"""
Path utility module for Tamil Nadu satellite datasets.
Resolves local Linux paths for reference tiles, temporal tiles, and metadata.
Guarantees correct path resolution independent of OS or legacy Windows paths in metadata files.
"""

from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_BASE = PROJECT_ROOT / "dataset_tamilnadu_districts"
TEMPORAL_BASE = PROJECT_ROOT / "data" / "tamilnadu_temporal"


def get_tile_image_path(district_key: str, tile_id: str) -> Path:
    """Return the absolute path to the reference (2024) tile image."""
    return DATASET_BASE / district_key / "close_aerial_z14" / f"{tile_id}.png"


def get_tile_meta_path(district_key: str, tile_id: str) -> Path:
    """Return the absolute path to the reference tile metadata JSON."""
    return DATASET_BASE / district_key / "close_aerial_z14" / f"{tile_id}_meta.json"


def get_temporal_image_path(district_key: str, tile_id: str, year: Union[str, int]) -> Path:
    """Return the absolute path to a temporal tile image (e.g., year 2020 or 2025)."""
    return TEMPORAL_BASE / district_key / f"year_{year}" / f"{tile_id}_{year}.png"


def get_temporal_meta_path(district_key: str, tile_id: str, year: Union[str, int]) -> Path:
    """Return the absolute path to a temporal tile metadata JSON."""
    return TEMPORAL_BASE / district_key / f"year_{year}" / f"{tile_id}_{year}_meta.json"


def resolve_file_path(path_str: Optional[str], district_key: Optional[str] = None, tile_id: Optional[str] = None) -> Optional[Path]:
    """
    Safely resolve a file path string. If it contains Windows backslashes or refers
    to an inaccessible path, attempt to reconstruct it from district_key and tile_id.
    """
    if path_str:
        p = Path(path_str)
        if p.exists():
            return p
        clean_str = path_str.replace("\\", "/")
        p_clean = Path(clean_str)
        if p_clean.exists():
            return p_clean

    if district_key and tile_id:
        p_recon = get_tile_image_path(district_key, tile_id)
        if p_recon.exists():
            return p_recon

    return None
