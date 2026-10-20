import sqlite3
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

DB_PATH = Path("data/change_regions.db")
SUPPORTED_CLASSES = ["road", "building", "water", "vegetation", "bare_land", "unknown"]
SUPPORTED_CHANGE_TYPES = ["new", "removed", "expanded", "transition", "unknown"]

class RegionDatabase:
    """SQLite Database manager for indexing and querying detected change regions."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS change_regions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    region_id INTEGER NOT NULL,
                    semantic_class TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    area_pixels INTEGER NOT NULL,
                    area_m2 REAL,
                    change_score REAL NOT NULL,
                    quality_score REAL,
                    confidence_category TEXT,
                    centroid_x INTEGER NOT NULL,
                    centroid_y INTEGER NOT NULL,
                    centroid_lat REAL,
                    centroid_lon REAL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_width INTEGER NOT NULL,
                    bbox_height INTEGER NOT NULL,
                    t1_class TEXT NOT NULL,
                    t2_class TEXT NOT NULL
                )
            """)
            # Safe column migrations if database already existed from Day 11
            for col, col_type in [("area_m2", "REAL"), ("quality_score", "REAL"),
                                  ("confidence_category", "TEXT"), ("centroid_lat", "REAL"),
                                  ("centroid_lon", "REAL")]:
                try:
                    cursor.execute(f"ALTER TABLE change_regions ADD COLUMN {col} {col_type}")
                except Exception:
                    pass

            # Create indexes for fast filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_class ON change_regions(semantic_class)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_change_type ON change_regions(change_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scene_id ON change_regions(scene_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON change_regions(run_id)")
            conn.commit()

    def insert_regions(self, run_id: str, scene_id: str, regions: List[Dict[str, Any]]):
        """Insert or replace region metadata for a given run and scene."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete existing records for run_id to avoid duplication
            cursor.execute("DELETE FROM change_regions WHERE run_id = ?", (run_id,))

            for r in regions:
                if not isinstance(r, dict):
                    continue
                semantic_class = str(r.get("Class", r.get("semantic_class", "unknown"))).lower().replace(" ", "_")
                change_type = str(r.get("Change Type", r.get("change_type", "unknown"))).lower().replace(" ", "_")
                t1_cls = str(r.get("T1 Class", r.get("t1_class", "unknown"))).lower().replace(" ", "_")
                t2_cls = str(r.get("T2 Class", r.get("t2_class", "unknown"))).lower().replace(" ", "_")
                area = int(r.get("Area (pixels)", r.get("Area", r.get("area_pixels", 0))))
                area_m2 = float(r.get("area_m2", area * 100.0)) # Default 10m x 10m = 100 m2/pixel

                # Extract proper probability confidence score (0.0 to 1.0)
                if "Mean Probability" in r and float(r["Mean Probability"]) > 0:
                    score = float(r["Mean Probability"])
                elif "confidence_percentage" in r and float(r["confidence_percentage"]) > 0:
                    score = float(r["confidence_percentage"]) / 100.0
                elif "change_score" in r and 0.0 < float(r["change_score"]) <= 1.0:
                    score = float(r["change_score"])
                elif "Significance Score" in r and area > 0:
                    score = min(1.0, max(0.1, float(r["Significance Score"]) / (area / 100.0)))
                else:
                    score = float(r.get("change_score", 0.75))
                score = round(float(min(1.0, max(0.0, score))), 4)

                quality_score = float(r.get("quality_score", round(score * min(1.0, area / 100.0), 2)))
                
                # Confidence Category rule
                if score >= 0.80 and area >= 100:
                    conf_cat = "HIGH_CONFIDENCE_CHANGE"
                elif score >= 0.65:
                    conf_cat = "MEDIUM_CONFIDENCE_CHANGE"
                else:
                    conf_cat = "LOW_CONFIDENCE_CHANGE"

                # Extract Bounding Box
                if "Bounding Box [X,Y,W,H]" in r:
                    bbox = r["Bounding Box [X,Y,W,H]"]
                elif "bbox" in r:
                    bbox = r["bbox"]
                else:
                    bbox = [r.get("X", 0), r.get("Y", 0), r.get("Width", 0), r.get("Height", 0)]

                # Extract Centroid
                if "Centroid [CX,CY]" in r:
                    centroid = r["Centroid [CX,CY]"]
                elif "centroid" in r:
                    centroid = r["centroid"]
                else:
                    centroid = [r.get("Centroid X", r.get("CX", 0)), r.get("Centroid Y", r.get("CY", 0))]

                lat = r.get("latitude", r.get("centroid_lat"))
                lon = r.get("longitude", r.get("centroid_lon"))

                region_id = int(r.get("Region ID", r.get("region_id", r.get("id", 0))))

                cursor.execute("""
                    INSERT INTO change_regions (
                        run_id, scene_id, region_id, semantic_class, change_type,
                        area_pixels, area_m2, change_score, quality_score, confidence_category,
                        centroid_x, centroid_y, centroid_lat, centroid_lon,
                        bbox_x, bbox_y, bbox_width, bbox_height, t1_class, t2_class
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, scene_id, region_id,
                    semantic_class, change_type, area, area_m2, score, quality_score, conf_cat,
                    int(centroid[0]), int(centroid[1]), lat, lon,
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    t1_cls, t2_cls
                ))
            conn.commit()

    def query_regions(self,
                      semantic_class: Optional[str] = None,
                      change_type: Optional[str] = None,
                      scene_id: Optional[str] = None,
                      district: Optional[str] = None,
                      min_area: Optional[int] = None,
                      max_area: Optional[int] = None,
                      min_score: Optional[float] = None,
                      sort_by: str = "area_desc",
                      limit: int = 50) -> List[Dict[str, Any]]:
        """Query indexed region database with structured parameters and exact location metadata."""
        def _execute_query(sem_cls_filter):
            q_str = "SELECT * FROM change_regions WHERE 1=1"
            q_params = []

            if sem_cls_filter and sem_cls_filter != "all":
                q_str += " AND semantic_class = ?"
                q_params.append(sem_cls_filter.lower().replace(" ", "_"))

            if change_type and change_type != "all":
                q_str += " AND change_type = ?"
                q_params.append(change_type.lower().replace(" ", "_"))

            if scene_id and scene_id != "all":
                q_str += " AND scene_id = ?"
                q_params.append(scene_id)

            if district and district != "all":
                q_str += " AND LOWER(scene_id) LIKE ?"
                q_params.append(f"{district.lower().strip()}%")

            if min_area is not None:
                q_str += " AND area_pixels >= ?"
                q_params.append(min_area)

            if max_area is not None:
                q_str += " AND area_pixels <= ?"
                q_params.append(max_area)

            if min_score is not None:
                q_str += " AND change_score >= ?"
                q_params.append(min_score)

            if sort_by == "area_desc":
                q_str += " ORDER BY area_pixels DESC"
            elif sort_by == "score_desc":
                q_str += " ORDER BY change_score DESC"
            elif sort_by == "region_asc":
                q_str += " ORDER BY id ASC"

            q_str += f" LIMIT {int(limit)}"

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(q_str, q_params)
                return cursor.fetchall()

        rows = _execute_query(semantic_class)

        # Smart fallback: if specific class yielded 0 results in the target district, relax class to find actual changes in that district
        if len(rows) == 0 and district and district != "all" and semantic_class and semantic_class != "all":
            rows = _execute_query("all")

        results = []
        for row in rows:
            r_dict = dict(row)
            lat = r_dict.get("centroid_lat")
            lon = r_dict.get("centroid_lon")
            sc_id = r_dict.get("scene_id", "")
            dist_name = sc_id.split("_")[0].capitalize() if "_" in sc_id else "Tamil Nadu"
            reg_id = r_dict.get("region_id", 0)
            run_id = r_dict.get("run_id", "")

            map_url = f"/map?lat={lat}&lon={lon}&zoom=16&region_id={reg_id}&district={dist_name.lower()}" if (lat and lon) else f"/map?scene_id={sc_id}"
            google_maps_url = f"https://www.google.com/maps?q={lat},{lon}" if (lat and lon) else None

            results.append({
                "id": r_dict.get("id"),
                "run_id": run_id,
                "scene_id": sc_id,
                "region_id": reg_id,
                "district": dist_name,
                "semantic_class": r_dict.get("semantic_class"),
                "change_type": r_dict.get("change_type"),
                "area_pixels": r_dict.get("area_pixels", 0),
                "area_m2": r_dict.get("area_m2", r_dict.get("area_pixels", 0) * 100.0),
                "change_score": round(r_dict.get("change_score", 0.0), 4),
                "quality_score": r_dict.get("quality_score", 0.0),
                "confidence_category": r_dict.get("confidence_category", "CHANGE"),
                "centroid": [r_dict.get("centroid_x", 0), r_dict.get("centroid_y", 0)],
                "centroid_lat": lat,
                "centroid_lon": lon,
                "bbox": [r_dict.get("bbox_x", 0), r_dict.get("bbox_y", 0), r_dict.get("bbox_width", 0), r_dict.get("bbox_height", 0)],
                "t1_class": r_dict.get("t1_class", "unknown"),
                "t2_class": r_dict.get("t2_class", "unknown"),
                "crop_url": f"/static/generated/{run_id}/crops/region_{reg_id}_crop.png",
                "exact_location": {
                    "latitude": round(lat, 5) if lat else None,
                    "longitude": round(lon, 5) if lon else None,
                    "coordinates_str": f"{lat:.5f}° N, {lon:.5f}° E" if (lat and lon) else "Grid-referenced",
                    "district": dist_name,
                    "map_url": map_url,
                    "google_maps_url": google_maps_url
                }
            })
        return results


class QueryParser:
    """Rule-based natural language parser mapping search phrases to structured query filters and geographic locations."""

    DISTRICTS = [
        "ariyalur", "chengalpattu", "chennai", "coimbatore", "cuddalore",
        "dharmapuri", "dindigul", "erode", "kallakurichi", "kanchipuram",
        "kanyakumari", "karur", "krishnagiri", "madurai", "mayiladuthurai",
        "nagapattinam", "namakkal", "nilgiris", "perambalur", "pudukkottai",
        "ramanathapuram", "ranipet", "salem", "sivaganga", "tenkasi",
        "thanjavur", "theni", "thoothukudi", "tiruchirappalli", "trichy", "tirunelveli",
        "tirupathur", "tiruppur", "tiruvallur", "tiruvannamalai", "tiruvarur",
        "vellore", "viluppuram", "virudhunagar"
    ]

    @staticmethod
    def parse_query(user_query: str) -> Dict[str, Any]:
        q = user_query.strip().lower()
        if not q:
            return {"status": "empty", "intent": {}, "message": "Empty query."}

        intent = {
            "semantic_class": "all",
            "change_type": "all",
            "district": None,
            "min_area": None,
            "min_score": None,
            "sort_by": "area_desc",
            "unsupported_term": None
        }

        # 1. Extract District / Geographic Location
        for dist in QueryParser.DISTRICTS:
            pattern = r'\b' + re.escape(dist) + r'\b'
            if re.search(pattern, q):
                intent["district"] = "tiruchirappalli" if dist == "trichy" else dist
                break

        # 2. Check for truly unsupported non-remote-sensing domains (e.g. airport, solar, aircraft)
        unsupported_keywords = ["airport", "solar", "aircraft", "satellite orbit"]
        for term in unsupported_keywords:
            if term in q:
                intent["unsupported_term"] = term
                return {
                    "status": "unsupported",
                    "intent": intent,
                    "message": f"'{term.capitalize()}' is not currently a supported target. Supported classes: Road, Building, Water, Vegetation, Bare Land, Unknown."
                }

        # 3. Extract Semantic Class (with natural language synonyms)
        # Prioritize explicit natural and specific entities (water, vegetation, road, bare land)
        # so contextual modifiers like 'near urban development' or 'in city' do not misclassify water as building
        if re.search(r"\bwater\b|\blake(s)?\b|\bpond(s)?\b|\breservoir(s)?\b|\briver(s)?\b|\bcanal(s)?\b|\bwater body\b|\bwaterbodies\b|\bocean\b|\bsea\b|\bwetland(s)?\b", q):
            intent["semantic_class"] = "water"
        elif re.search(r"\bvegetation\b|\bcrop(s)?\b|\bgreenery\b|\btree(s)?\b|\bforest(s)?\b|\bfarm(land)?\b|\bagricultur(e|al)\b", q):
            intent["semantic_class"] = "vegetation"
        elif re.search(r"\broad(s)?\b|\bstreet(s)?\b|\bhighway(s)?\b|\bcorridor\b|\bexpressway\b", q):
            intent["semantic_class"] = "road"
        elif re.search(r"\bbare\b|\bsoil\b|\bground\b|\bcleared\b|\bexcavation\b|\bquarry\b|\bmine\b", q):
            intent["semantic_class"] = "bare_land"
        elif re.search(r"\bbuilding(s)?\b|\bstructure(s)?\b|\bbuilt-up\b|\bhouse(s)?\b|\bconstruction\b|\burban\b|\bcity\b", q):
            intent["semantic_class"] = "building"

        # 4. Extract Change Type
        if re.search(r"\bnew\b|\bcreated\b|\bappeared\b", q):
            intent["change_type"] = "new"
        elif re.search(r"\bremoved\b|\bcleared\b|\bdemolished\b|\blost\b|\bloss\b|\bdisappeared\b", q):
            intent["change_type"] = "removed"
        elif re.search(r"\bexpand(ed)?\b|\bwidened\b|\bgrown\b|\bgrowth\b|\bextension\b", q):
            intent["change_type"] = "expanded"
        elif re.search(r"\btransition(ed)?\b|\bshifted\b|\bconverted\b|\bencroachment\b", q):
            intent["change_type"] = "transition"

        # 5. Extract Area / Size Modifiers
        if re.search(r"\blarge\b|\bbig\b|\bmajor\b|\bsignificant\b|\bextensive\b", q):
            intent["min_area"] = 200
            intent["sort_by"] = "area_desc"
        elif re.search(r"\bsmall\b|\bminor\b|\btiny\b", q):
            intent["sort_by"] = "area_asc"

        # 6. Extract Confidence / Probability Modifiers
        if re.search(r"\bhigh confidence\b|\bhigh probability\b|\bconfident\b|\bcertain\b", q):
            intent["min_score"] = 0.75
            intent["sort_by"] = "score_desc"

        return {
            "status": "success",
            "query": user_query,
            "intent": intent,
            "message": "Query parsed successfully."
        }
