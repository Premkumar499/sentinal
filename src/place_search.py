"""
Place Name Search Engine
Integrates the place index with the existing query system.

Allows queries like:
  "thanthonimalai"  →  finds tile(s), district, GPS coords
  "gandhigramam karur" → filters by district too
  "new buildings near gandhigramam 2020 to 2024" → parsed by existing QueryParser
"""

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLACE_INDEX_DIR = PROJECT_ROOT / "data" / "place_index"

_place_lookup: Optional[Dict] = None
_tile_places: Optional[Dict] = None
_tile_sequence: Optional[Dict] = None


def _load():
    global _place_lookup, _tile_places, _tile_sequence
    if _place_lookup is not None:
        return

    place_path = PLACE_INDEX_DIR / "place_lookup.json"
    tile_path  = PLACE_INDEX_DIR / "tile_places.json"
    seq_path   = PLACE_INDEX_DIR / "tile_sequence.json"

    _place_lookup  = json.loads(place_path.read_text(encoding="utf-8")) if place_path.exists() else {}
    _tile_places   = json.loads(tile_path.read_text(encoding="utf-8"))  if tile_path.exists()  else {}
    _tile_sequence = json.loads(seq_path.read_text(encoding="utf-8"))   if seq_path.exists()   else {}


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


def is_ready() -> bool:
    """Returns True if place index files exist on disk."""
    return (PLACE_INDEX_DIR / "place_lookup.json").exists()


def search_place(query: str, district: Optional[str] = None,
                 top_k: int = 10) -> List[Dict]:
    """
    Search for a place name. Returns list of matching places with
    tile IDs, GPS coords, and district.

    Args:
        query    : place name or partial name
        district : optional district filter
        top_k    : max results

    Returns list of:
      {name, district, lat, lon, place_type, tiles, years_available,
       similarity, tile_sequence_positions}
    """
    _load()
    if not _place_lookup:
        return []

    q = _normalize(query)
    results = []

    for key, entry in _place_lookup.items():
        # Substring match — "gandhi" matches "gandhigramam"
        if q not in key and key not in q:
            continue

        # District filter
        if district and district.lower() not in ["all", ""]:
            if _normalize(district) not in _normalize(entry.get("district", "")):
                continue

        # Compute a simple relevance score — exact match scores highest
        if key == q:
            score = 1.0
        elif key.startswith(q) or q.startswith(key):
            score = 0.85
        else:
            score = 0.6

        # Add tile sequence positions (N→S index within district)
        dist_key = entry.get("district", "")
        seq = _tile_sequence.get(dist_key, []) if _tile_sequence else []
        positions = [seq.index(t) + 1 for t in entry.get("tiles", []) if t in seq]

        results.append({
            "name":               entry["name"],
            "name_normalized":    key,
            "district":           dist_key,
            "lat":                entry.get("lat", 0.0),
            "lon":                entry.get("lon", 0.0),
            "place_type":         entry.get("place_type", "place"),
            "tiles":              entry.get("tiles", []),
            "years_available":    entry.get("years_available", ["2020", "2022", "2024"]),
            "similarity":         round(score, 2),
            "tile_sequence_pos":  positions,
            "map_url": (
                f"/map?lat={entry.get('lat', 0)}&lon={entry.get('lon', 0)}"
                f"&zoom=15&district={dist_key}"
            ) if entry.get("lat") else f"/map?district={dist_key}"
        })

    # Sort by score then name length (shorter = more exact)
    results.sort(key=lambda x: (-x["similarity"], len(x["name"])))
    return results[:top_k]


def get_tiles_for_place(place_name: str,
                         district: Optional[str] = None) -> List[str]:
    """Return list of tile_ids covering a place name."""
    matches = search_place(place_name, district=district, top_k=1)
    if matches:
        return matches[0].get("tiles", [])
    return []


def get_places_in_tile(tile_id: str) -> List[Dict]:
    """Return all named places inside a given tile."""
    _load()
    if not _tile_places:
        return []
    tile_data = _tile_places.get(tile_id, {})
    return tile_data.get("places", [])


def get_district_sequence(district: str) -> List[str]:
    """
    Return tiles for a district ordered geographically N→S, W→E.
    Useful for walking a district tile by tile.
    """
    _load()
    if not _tile_sequence:
        return []
    return _tile_sequence.get(district.lower(), [])


def get_neighboring_tiles(tile_id: str, district: str) -> Dict[str, Optional[str]]:
    """
    Given a tile_id, return its geographic neighbours
    (north, south, east, west) in the district sequence grid.
    """
    _load()
    seq = _tile_sequence.get(district.lower(), []) if _tile_sequence else []
    if tile_id not in seq:
        return {"north": None, "south": None, "east": None, "west": None}

    # Parse row/col from tile_id e.g. KARUR_CLOSE_Z14_R03_C05
    import re
    m = re.search(r"R(\d+)_C(\d+)", tile_id)
    if not m:
        return {"north": None, "south": None, "east": None, "west": None}

    row, col = int(m.group(1)), int(m.group(2))
    prefix = tile_id[:m.start()]

    def find(r, c):
        tid = f"{prefix}R{r:02d}_C{c:02d}"
        return tid if tid in seq else None

    return {
        "north": find(row - 1, col),
        "south": find(row + 1, col),
        "west":  find(row, col - 1),
        "east":  find(row, col + 1),
    }


def stats() -> Dict:
    """Return index statistics."""
    _load()
    return {
        "ready":          is_ready(),
        "total_places":   len(_place_lookup) if _place_lookup else 0,
        "total_tiles":    len(_tile_places)  if _tile_places  else 0,
        "districts":      len(_tile_sequence) if _tile_sequence else 0,
    }
