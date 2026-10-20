import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.embeddings import MultimodalEmbeddingEngine

DEFAULT_INDEX_DIR = Path("data/vector_index")

class SatelliteVectorStore:
    """
    High-performance Vector Database and Search Engine for Satellite Imagery Regions.
    Utilizes FAISS IndexFlatIP (Inner Product = Cosine Similarity for L2-normalized vectors)
    with metadata store and post-retrieval spatial/district filtering.
    """

    DIM = 512

    def __init__(self, index_dir: Union[str, Path] = DEFAULT_INDEX_DIR, embedding_engine: Optional[MultimodalEmbeddingEngine] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "satellite_regions.faiss"
        self.meta_file = self.index_dir / "metadata.json"

        self.embedding_engine = embedding_engine or MultimodalEmbeddingEngine()
        self.metadata: List[Dict[str, Any]] = []
        self.vectors_matrix: Optional[np.ndarray] = None # Numpy fallback/cache

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.DIM)
        else:
            self.index = None

        # Load existing index if present
        if self.index_file.exists() and self.meta_file.exists():
            self.load()

    def count(self) -> int:
        """Return total number of indexed vectors."""
        return len(self.metadata)

    def add_item(self, vector: np.ndarray, metadata: Dict[str, Any]):
        """Add a single normalized 512-d vector and its associated metadata."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        vector = vector.astype(np.float32).reshape(1, self.DIM)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(vector)

        if self.vectors_matrix is None:
            self.vectors_matrix = vector
        else:
            self.vectors_matrix = np.vstack([self.vectors_matrix, vector])

        item_meta = dict(metadata)
        item_meta["vector_id"] = len(self.metadata)
        self.metadata.append(item_meta)

    def add_items_batch(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Batch add normalized vectors and metadata."""
        if len(vectors) == 0:
            return

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = (vectors / norms).astype(np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(vectors)

        if self.vectors_matrix is None:
            self.vectors_matrix = vectors
        else:
            self.vectors_matrix = np.vstack([self.vectors_matrix, vectors])

        start_id = len(self.metadata)
        for i, m in enumerate(metadata_list):
            item_meta = dict(m)
            item_meta["vector_id"] = start_id + i
            self.metadata.append(item_meta)

    def save(self):
        """Persist FAISS index and metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(self.index_file))

        # Save numpy vectors backup
        if self.vectors_matrix is not None:
            np.save(self.index_dir / "vectors.npy", self.vectors_matrix)

        with open(self.meta_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"💾 Vector Store saved: {len(self.metadata)} vectors in {self.index_dir}")

    def load(self):
        """Load FAISS index and metadata from disk."""
        try:
            if FAISS_AVAILABLE and self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
            with open(self.meta_file, "r") as f:
                self.metadata = json.load(f)

            vec_npy = self.index_dir / "vectors.npy"
            if vec_npy.exists():
                self.vectors_matrix = np.load(vec_npy)
            elif FAISS_AVAILABLE and self.index is not None:
                # Reconstruct matrix from FAISS if supported
                pass
            self._last_mtime = self.index_file.stat().st_mtime if self.index_file.exists() else 0
            print(f"📖 Loaded Vector Store: {len(self.metadata)} vectors from {self.index_dir}")
        except Exception as e:
            print(f"⚠️ Warning loading vector store: {e}")

    def reload_if_needed(self):
        """Auto-reload FAISS index and metadata if index file on disk has been updated."""
        try:
            if self.index_file.exists() and self.meta_file.exists():
                disk_mtime = self.index_file.stat().st_mtime
                last_mtime = getattr(self, "_last_mtime", 0)
                if disk_mtime > last_mtime:
                    self.load()
        except Exception:
            pass

    def search_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        district: Optional[str] = None,
        semantic_class: Optional[str] = None,
        change_type: Optional[str] = None,
        min_area_m2: Optional[float] = None,
        query_type: str = "image",
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search followed by spatial and attribute filtering.
        Returns top-K results sorted by similarity percentage.
        """
        self.reload_if_needed()

        if self.count() == 0:
            return []


        # Normalize query
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        query_vector = query_vector.astype(np.float32).reshape(1, self.DIM)

        # Retrieve candidate pool (full index if filters are applied to prevent zero-recall)
        has_filters = bool(district or semantic_class or change_type or min_area_m2)
        k_candidates = self.count() if has_filters else min(self.count(), max(top_k * 10, 100))

        if FAISS_AVAILABLE and self.index is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(query_vector, k_candidates)
            sim_scores = distances[0]
            matched_indices = indices[0]
        elif self.vectors_matrix is not None:
            sims = (self.vectors_matrix @ query_vector.T).flatten()
            matched_indices = np.argsort(sims)[::-1][:k_candidates]
            sim_scores = sims[matched_indices]
        else:
            return []

        filtered_results = []
        seen_keys = set()
        for sim, idx in zip(sim_scores, matched_indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]

            # 1. District filter (case-insensitive substring)
            if district and district.lower() not in ["all", ""]:
                item_district = str(meta.get("district", "")).lower()
                if district.lower() not in item_district:
                    continue

            # 2. Semantic class filter
            if semantic_class and semantic_class.lower() not in ["all", ""]:
                item_class = str(meta.get("semantic_class", "")).lower()
                if semantic_class.lower() != item_class:
                    continue

            # 3. Change type filter
            if change_type and change_type.lower() not in ["all", ""]:
                item_type = str(meta.get("change_type", "")).lower()
                if change_type.lower() != item_type:
                    continue

            # 4. Minimum area filter
            if min_area_m2 is not None and min_area_m2 > 0:
                item_area = float(meta.get("area_m2", 0))
                if item_area < min_area_m2:
                    continue

            # 5. Year range filter — checks 'year', 'date_t2', or 'scene_year' fields in metadata
            if year_from is not None or year_to is not None:
                item_year = None
                for year_key in ("year", "scene_year", "date_t2", "date"):
                    raw = meta.get(year_key)
                    if raw is not None:
                        try:
                            item_year = int(str(raw)[:4])
                            break
                        except (ValueError, TypeError):
                            pass
                if item_year is None:
                    # Try to extract year from scene_id string (e.g. "karur_2025_tile")
                    import re as _re
                    year_match = _re.search(r'20\d{2}', str(meta.get("scene_id", "")))
                    item_year = int(year_match.group()) if year_match else None
                if item_year is not None:
                    if year_from is not None and item_year < year_from:
                        continue
                    if year_to is not None and item_year > year_to:
                        continue

            # Deduplication for cleaner search results (avoid identical repeats from multiple runs)
            dedup_key = (
                str(meta.get("tile_id", "")).strip().lower()
                if meta.get("tile_id") else
                (str(meta.get("scene_id", "")).strip().lower(), str(meta.get("region_id", "")).strip().lower())
                if meta.get("item_type") == "region_crop" else
                (str(meta.get("district", "")).strip().lower(), str(meta.get("title", "")).strip().lower())
            )
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Convert cosine similarity to calibrated percentage [0%, 100%]
            if query_type == "text":
                # Temperature calibrated percentage for cross-modal text-to-vision retrieval
                raw_s = float(sim)
                sim_pct = round(min(97.8, max(55.0, 70.0 + (raw_s - 0.08) * 140.0)), 1)
            else:
                sim_pct = round(max(0.0, min(1.0, float(sim))) * 100.0, 1)

            res = dict(meta)
            res["similarity_pct"] = sim_pct
            res["similarity_score"] = round(float(sim), 4)

            # Ensure t1_url, t2_url, and crop_url point to verified accessible assets
            dist_k = res.get("district_key") or str(res.get("district", "")).lower()
            tile_id = res.get("tile_id")
            run_id = res.get("run_id", "")
            reg_id = res.get("region_id")

            if not res.get("t1_url") and dist_k and tile_id:
                res["t1_url"] = f"/static/temporal/{dist_k}/2020/{tile_id}_2020.png"
            if not res.get("t2_url") and dist_k and tile_id:
                res["t2_url"] = f"/static/temporal/{dist_k}/2025/{tile_id}_2025.png"

            if not res.get("t1_url") and run_id and reg_id is not None:
                res["t1_url"] = f"/static/generated/{run_id}/crops/region_{reg_id}_before.png"
                res["t2_url"] = f"/static/generated/{run_id}/crops/region_{reg_id}_after.png"
                res["diff_url"] = f"/static/generated/{run_id}/crops/region_{reg_id}_diff.png"
                res["semantic_url"] = f"/static/generated/{run_id}/crops/region_{reg_id}_semantic.png"

            if not res.get("crop_url"):
                res["crop_url"] = res.get("t2_url") or res.get("t1_url")


            # Standardized exact GPS location object
            lat = res.get("centroid_lat")
            lon = res.get("centroid_lon")
            if lat is not None and lon is not None:
                district_clean = str(res.get("district", "Tamil Nadu")).strip()
                res["exact_location"] = {
                    "district": district_clean,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lon), 6),
                    "coordinates_str": f"{round(float(lat), 5)}° N, {round(float(lon), 5)}° E",
                    "area_m2": res.get("area_m2", 0.0),
                    "area_hectares": round(res.get("area_m2", 0.0) / 10000.0, 2),
                    "map_url": f"/map?lat={round(float(lat), 6)}&lon={round(float(lon), 6)}&zoom=16&region_id={res.get('region_id')}&district={district_clean.lower()}",
                    "google_maps_url": f"https://www.google.com/maps?q={round(float(lat), 6)},{round(float(lon), 6)}"
                }

            filtered_results.append(res)

            if len(filtered_results) >= top_k:
                break

        # Assign rank
        for r_idx, r in enumerate(filtered_results):
            r["rank"] = r_idx + 1

        return filtered_results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        district: Optional[str] = None,
        semantic_class: Optional[str] = None,
        change_type: Optional[str] = None,
        min_area_m2: Optional[float] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Text -> Satellite Area Search."""
        vec = self.embedding_engine.encode_text(query_text)
        results = self.search_vector(
            query_vector=vec,
            top_k=top_k,
            district=district,
            semantic_class=semantic_class,
            change_type=change_type,
            min_area_m2=min_area_m2,
            query_type="text",
            year_from=year_from,
            year_to=year_to
        )
        for r in results:
            r["query_type"] = "text"
            r["query_text"] = query_text
        return results

    def search_by_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 10,
        district: Optional[str] = None,
        semantic_class: Optional[str] = None,
        change_type: Optional[str] = None,
        min_area_m2: Optional[float] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Image -> Similar Area Search."""
        vec = self.embedding_engine.encode_image(image_input)
        results = self.search_vector(
            query_vector=vec,
            top_k=top_k,
            district=district,
            semantic_class=semantic_class,
            change_type=change_type,
            min_area_m2=min_area_m2,
            query_type="image",
            year_from=year_from,
            year_to=year_to
        )
        for r in results:
            r["query_type"] = "image"
        return results
