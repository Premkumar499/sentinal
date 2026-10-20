#!/usr/bin/env python3
"""
High-Speed Visual Vector Database Builder for Satellite Image RAG Search.
Uses multi-threaded PIL decoding + PyTorch ResNet-18 (512-d unit sphere)
to index genuine Sentinel-2/Wayback bi-temporal satellite tiles across Tamil Nadu.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.embeddings import MultimodalEmbeddingEngine
from src.vector_store import SatelliteVectorStore

def process_single_tile(item):
    """Load image once, downscale, extract fast spectral class, and prepare RGB tensor."""
    pair, transform_fn = item
    img_path = pair["year_2025"]["image_path"]
    try:
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

        # Quick spectral class from 224x224 RGB
        arr = np.asarray(img, dtype=np.float32) / 255.0
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        mr, mg, mb = np.mean(r), np.mean(g), np.mean(b)
        green_idx = (mg - mr) / (mg + mr + 1e-6)
        contrast = np.std(arr)
        brightness = (mr + mg + mb) / 3.0

        if mb > mr and mb > 0.30 and contrast < 0.12:
            sem_class = "water_reservoir"
        elif green_idx > 0.03 and mg > mb:
            sem_class = "vegetation"
        elif contrast > 0.16 and brightness > 0.35:
            sem_class = "dense_urban"
        elif mr > mb and mr > 0.32:
            sem_class = "bare_land"
        else:
            sem_class = "mixed_terrain"

        tensor = transform_fn(img)
        return (pair, tensor, sem_class, None)
    except Exception as e:
        return (pair, None, "mixed_terrain", str(e))

def build_index():
    print("=" * 80)
    print("🛰️  HIGH-SPEED MULTI-THREADED SATELLITE VECTOR INDEXER")
    print("=" * 80)

    catalog_path = project_root / "data" / "temporal_catalog_2020_2025" / "master_catalog_2020_2025.json"
    if not catalog_path.exists():
        print(f"❌ Catalog not found: {catalog_path}")
        return

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    pairs = catalog.get("pairs", [])
    valid_pairs = []
    for p in pairs:
        y25 = p.get("year_2025", {}).get("image_path")
        y20 = p.get("year_2020", {}).get("image_path")
        if y25 and y20 and os.path.exists(y25) and os.path.exists(y20):
            valid_pairs.append(p)

    print(f"Total pairs in catalog: {len(pairs)}")
    print(f"Valid bi-temporal pairs on disk: {len(valid_pairs)}")

    engine = MultimodalEmbeddingEngine()
    store = SatelliteVectorStore(index_dir="data/vector_index", embedding_engine=engine)
    # Clear existing to build fresh clean index
    store.metadata = []
    if store.index is not None:
        store.index.reset()
    store.vectors_matrix = None

    import torch
    batch_size = 128
    total_valid = len(valid_pairs)
    start_time = time.time()

    print(f"Starting parallel extraction with 8 loader threads, batch size {batch_size}...")

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i in range(0, total_valid, batch_size):
            chunk = valid_pairs[i:i + batch_size]
            task_inputs = [(p, engine.transform) for p in chunk]

            results = list(pool.map(process_single_tile, task_inputs))

            valid_tensors = []
            meta_list = []
            for pair, tensor, sem_class, err in results:
                if tensor is None:
                    continue
                valid_tensors.append(tensor)

                dist_key = pair.get("district_key", "tamilnadu")
                dist_name = pair.get("district_name", dist_key.title())
                tile_id = pair.get("tile_id")
                lat = pair.get("center_lat", 11.1271)
                lon = pair.get("center_lon", 78.6569)
                area_m2 = round(pair.get("area_sq_km", 51.7) * 1_000_000, 1)

                t1_url = f"/static/temporal/{dist_key}/2020/{tile_id}_2020.png"
                t2_url = f"/static/temporal/{dist_key}/2025/{tile_id}_2025.png"

                meta = {
                    "item_type": "bitemporal_tile",
                    "tile_id": tile_id,
                    "district": dist_name,
                    "district_key": dist_key,
                    "centroid_lat": lat,
                    "centroid_lon": lon,
                    "area_m2": area_m2,
                    "semantic_class": sem_class,
                    "change_type": "bitemporal_pair",
                    "t1_url": t1_url,
                    "t2_url": t2_url,
                    "crop_url": t2_url,
                    "title": f"{dist_name} — {tile_id}",
                    "description": f"Sentinel-2 / Wayback bi-temporal observation in {dist_name} ({area_m2:,.0f} m²).",
                    "year": 2025,
                    "exact_location": {
                        "district": dist_name,
                        "latitude": round(lat, 6),
                        "longitude": round(lon, 6),
                        "coordinates_str": f"{round(lat, 4)}° N, {round(lon, 4)}° E",
                        "area_m2": area_m2,
                        "area_hectares": round(area_m2 / 10000.0, 1)
                    }
                }
                meta_list.append(meta)

            if valid_tensors:
                batch_tensor = torch.stack(valid_tensors)
                with torch.no_grad():
                    feats = engine.resnet(batch_tensor).cpu().numpy()
                norms = np.linalg.norm(feats, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = (feats / norms).astype(np.float32)

                store.add_items_batch(vectors, meta_list)

            elapsed = time.time() - start_time
            done = min(i + batch_size, total_valid)
            fps = done / max(elapsed, 0.001)
            eta = (total_valid - done) / max(fps, 0.001)
            print(f"[{done:4d}/{total_valid}] ({done/total_valid*100:5.1f}%) | Speed: {fps:5.1f} tiles/s | ETA: {eta:4.0f}s")

    store.save()
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"🎉 Successfully built Vector Store with {store.count()} bi-temporal satellite tiles in {total_time:.1f}s!")
    print(f"📁 Persisted in: data/vector_index/")
    print("=" * 80)

if __name__ == "__main__":
    build_index()
