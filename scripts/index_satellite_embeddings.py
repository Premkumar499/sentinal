import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.embeddings import MultimodalEmbeddingEngine
from src.vector_store import SatelliteVectorStore

def index_all_satellite_data():
    """Build persistent multimodal FAISS vector index of all satellite regions and tiles."""
    print("🚀 Initializing Satellite Vector Store Indexer...")
    embedding_engine = MultimodalEmbeddingEngine()
    vector_store = SatelliteVectorStore(index_dir="data/vector_index", embedding_engine=embedding_engine)

    # Load district centers for coordinate and district identification
    dist_config_path = project_root / "config" / "districts.json"
    district_centers = {}
    if dist_config_path.exists():
        with open(dist_config_path, "r", encoding="utf-8") as f:
            district_centers = json.load(f)

    # 1. Index Region Crops from outputs/results/
    results_dir = project_root / "outputs" / "results"
    indexed_count = 0

    print("🛰️ Scanning outputs/results for detected change regions and crops...")
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_file = run_dir / "region_summary.json"
        crops_dir = run_dir / "crops"

        if not summary_file.exists() or not crops_dir.exists():
            continue

        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
            regions = summary.get("regions", [])
            scene_id = summary.get("scene_id", run_dir.name)
            run_id = summary.get("run_id", run_dir.name)

            # Determine district dynamically from scene_id or run_id
            district = "Tamil Nadu"
            sid_lower = scene_id.lower()
            rid_lower = run_id.lower()
            for d_key, d_info in district_centers.items():
                if d_key != "all" and (d_key in sid_lower or d_key in rid_lower):
                    district = d_info.get("name", d_key.capitalize())
                    break

            for r in regions:
                reg_id = r.get("Region ID", r.get("region_id"))
                crop_path = crops_dir / f"region_{reg_id}_crop.png"
                if not crop_path.exists():
                    continue

                sem_class = str(r.get("Class", r.get("semantic_class", "unknown"))).lower().replace(" ", "_")
                chg_type = str(r.get("Change Type", r.get("change_type", "unknown"))).lower().replace(" ", "_")
                area_m2 = float(r.get("area_m2", r.get("Area", 100) * 100.0))
                default_center = district_centers.get(district.lower(), district_centers.get("all", {"lat": 11.1271, "lon": 78.6569}))
                lat = float(r.get("centroid_lat") or default_center.get("lat", 11.1271))
                lon = float(r.get("centroid_lon") or default_center.get("lon", 78.6569))

                # Encode crop with embedding engine
                vec = embedding_engine.encode_image(crop_path)

                # Relative URL for web display
                crop_url = f"/static/generated/{run_id}/crops/region_{reg_id}_crop.png"

                metadata = {
                    "item_type": "region_crop",
                    "region_id": reg_id,
                    "scene_id": scene_id,
                    "run_id": run_id,
                    "district": district,
                    "semantic_class": sem_class,
                    "change_type": chg_type,
                    "area_m2": area_m2,
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "crop_url": crop_url,
                    "title": f"{district} — Region #{reg_id} ({sem_class.title()})",
                    "description": f"Detected {chg_type} of {sem_class} ({area_m2:,.1f} m²) in {district}."
                }

                vector_store.add_item(vec, metadata)
                indexed_count += 1

        except Exception as e:
            print(f"⚠️ Warning indexing {run_dir.name}: {e}")

    print(f"✅ Indexed {indexed_count} change region crops.")

    # 2. Index representative Tamil Nadu district tiles
    tn_dir = project_root / "dataset_tamilnadu_districts"
    tile_count = 0
    if tn_dir.exists():
        print("🗺️ Scanning Tamil Nadu district tiles for indexing...")
        for dist_folder in sorted(tn_dir.iterdir()):
            if not dist_folder.is_dir() or dist_folder.name.startswith("."):
                continue
            dist_name = dist_folder.name.capitalize()
            close_dir = dist_folder / "close_aerial_z14"
            if not close_dir.exists():
                continue

            for img_file in sorted(close_dir.glob("*.png"))[:3]: # Index up to 3 tiles per district for diversity
                meta_file = Path(str(img_file).replace(".png", "_meta.json"))
                lat, lon = 11.1271, 78.6569 # Default center of TN
                area_m2 = 768 * 768 * 100 # Approx tile ground area

                if meta_file.exists():
                    try:
                        with open(meta_file) as mf:
                            m_data = json.load(mf)
                        bbox = m_data.get("bbox", [])
                        if len(bbox) >= 4:
                            lon = round((bbox[0] + bbox[2]) / 2.0, 6)
                            lat = round((bbox[1] + bbox[3]) / 2.0, 6)
                    except Exception:
                        pass

                # Encode tile
                vec = embedding_engine.encode_image(img_file)
                tile_url = f"/static/tiles/{dist_folder.name}/{img_file.name}"

                metadata = {
                    "item_type": "district_tile",
                    "region_id": 0,
                    "scene_id": img_file.stem,
                    "run_id": "catalog",
                    "district": dist_name,
                    "semantic_class": "mixed",
                    "change_type": "baseline",
                    "area_m2": area_m2,
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "crop_url": tile_url,
                    "title": f"{dist_name} — High-Res Aerial Tile",
                    "description": f"Sentinel-2 / Wayback 10m high-resolution baseline tile for {dist_name} district."
                }

                vector_store.add_item(vec, metadata)
                tile_count += 1

    print(f"✅ Indexed {tile_count} Tamil Nadu district baseline tiles.")

    # Save to disk
    vector_store.save()
    print(f"🎉 Total items in Vector Store: {vector_store.count()}")
    print("Files written to data/vector_index/")

if __name__ == "__main__":
    index_all_satellite_data()
