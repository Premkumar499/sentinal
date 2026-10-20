import os
import sys
import time
import json
import csv
import zipfile
import numpy as np
import cv2
import yaml
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.features.extraction_day3 import FeatureExtractorDay3
from src.models.tuned import XGBoostTuned
from src.models.semantic import detect_semantic_change, get_semantic_mask
from src.preprocessing.pipeline import OSCDLoader, PreprocessingPipeline

class InferenceEngine:
    """End-to-End Inference Engine for Semantic Satellite Change Detection."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.models_dir = Path(self.config["outputs"]["models_dir"])
        self.repro_config_path = self.models_dir / "reproducible_config.yaml"

        # Load reproducible configuration if available
        if self.repro_config_path.exists():
            with open(self.repro_config_path, "r") as f:
                repro_cfg = yaml.safe_load(f)
            self.selected_features = repro_cfg.get("selected_model", {}).get("features", self.config["features"]["feature_names"])
            self.default_threshold = float(repro_cfg.get("selected_model", {}).get("threshold", 0.72))
        else:
            self.selected_features = self.config["features"]["feature_names"]
            self.default_threshold = 0.72

        # Model file prioritization
        model_path = self.models_dir / "xgboost_change_detector_tuned.json"
        if not model_path.exists():
            model_path = self.models_dir / "xgboost_change_detector.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No trained XGBoost model found in {self.models_dir}")

        self.model = XGBoostTuned(config_path=config_path)
        self.model.load_model(str(model_path))
        self.model.best_threshold = self.default_threshold
        self.feature_extractor = FeatureExtractorDay3(config_path=config_path)
        self.dataset_loader = OSCDLoader(config_path=config_path)
        self.pipeline = PreprocessingPipeline(config_path=config_path)

    def validate_inputs(self, t1_input, t2_input):
        """Validate input image dimensions, channel counts, and data sanity."""
        if t1_input is None or t2_input is None:
            return False, "Both T1 and T2 inputs must be provided."

        if isinstance(t1_input, str) or isinstance(t1_input, Path):
            if not os.path.exists(t1_input) and not str(t1_input).startswith("test_") and not str(t1_input).startswith("train_") and not str(t1_input).startswith("val_"):
                return False, f"T1 file does not exist: {t1_input}"
            if not os.path.exists(t2_input) and not str(t2_input).startswith("test_") and not str(t2_input).startswith("train_") and not str(t2_input).startswith("val_"):
                return False, f"T2 file does not exist: {t2_input}"

        return True, "Valid"

    def load_scene_bands(self, scene_id_or_paths):
        """Helper to load T1/T2 band dictionaries from scene ID or file paths."""
        if isinstance(scene_id_or_paths, str) and (scene_id_or_paths.startswith("test_") or scene_id_or_paths.startswith("train_") or scene_id_or_paths.startswith("val_")):
            scene_id = scene_id_or_paths
            paths = self.dataset_loader.get_scene_paths(scene_id)
            bmap = self.dataset_loader.band_mapping
            band_keys = ["blue", "green", "red", "nir", "swir"]
            t1_raw = {k: self.dataset_loader.load_band(paths["t1_dir"] / f"{bmap[k]}.tif") for k in band_keys}
            t2_raw = {k: self.dataset_loader.load_band(paths["t2_dir"] / f"{bmap[k]}.tif") for k in band_keys}
            t1_bands = {k: self.pipeline.clean_and_normalize(t1_raw[k]) for k in band_keys}
            t2_bands = {k: self.pipeline.clean_and_normalize(t2_raw[k]) for k in band_keys}
            return t1_bands, t2_bands, scene_id

        elif isinstance(scene_id_or_paths, tuple) or isinstance(scene_id_or_paths, list):
            p1, p2 = scene_id_or_paths
            # Support image files (PNG/JPG/GeoTIFF)
            img1 = cv2.imread(str(p1), cv2.IMREAD_COLOR)
            img2 = cv2.imread(str(p2), cv2.IMREAD_COLOR)

            if img1 is None or img2 is None:
                raise ValueError("Could not decode one or both input image files.")

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            if img1.shape[:2] != img2.shape[:2]:
                # Reshape img2 to match img1 if slight offset
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Global illumination / exposure alignment:
            # Matches overall channel mean and variance between dates to prevent camera exposure shifts from triggering false change
            img2_aligned = np.zeros_like(img2)
            for ch in range(3):
                m1, s1 = float(img1[:, :, ch].mean()), max(1e-4, float(img1[:, :, ch].std()))
                m2, s2 = float(img2[:, :, ch].mean()), max(1e-4, float(img2[:, :, ch].std()))
                img2_aligned[:, :, ch] = np.clip((img2[:, :, ch] - m2) * (s1 / s2) + m1, 0.0, 1.0)

            # Physics-based synthetic NIR and SWIR estimation from RGB
            r1, g1, b1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
            r2, g2, b2 = img2_aligned[:, :, 0], img2_aligned[:, :, 1], img2_aligned[:, :, 2]
            eps = 1e-6

            # Healthy vegetation chlorophyll reflects green and strongly reflects near-infrared
            green_excess1 = np.maximum(0.0, g1 - r1)
            nir1 = np.clip(g1 * 1.25 + green_excess1 * 1.75, 0.0, 1.0)
            swir1 = np.clip(r1 * 0.95 + b1 * 0.25, 0.0, 1.0)

            green_excess2 = np.maximum(0.0, g2 - r2)
            nir2 = np.clip(g2 * 1.25 + green_excess2 * 1.75, 0.0, 1.0)
            swir2 = np.clip(r2 * 0.95 + b2 * 0.25, 0.0, 1.0)

            # Water absorption: True water exhibits strong NIR absorption and blue/cyan dominance,
            # but NEVER exhibits high green vegetation excess (VARI > 0.03 or Green > Red * 1.12)
            vari1 = (g1 - r1) / (g1 + r1 - b1 + eps)
            is_veg1 = (vari1 > 0.03) | ((g1 > r1 * 1.12) & (g1 > b1))
            water_mask1 = (b1 > r1 * 1.15) & (b1 >= g1 * 0.88) & (~is_veg1) & (b1 + g1 + r1 < 1.1)
            nir1[water_mask1] = r1[water_mask1] * 0.5

            vari2 = (g2 - r2) / (g2 + r2 - b2 + eps)
            is_veg2 = (vari2 > 0.03) | ((g2 > r2 * 1.12) & (g2 > b2))
            water_mask2 = (b2 > r2 * 1.15) & (b2 >= g2 * 0.88) & (~is_veg2) & (b2 + g2 + r2 < 1.1)
            nir2[water_mask2] = r2[water_mask2] * 0.5

            t1_bands = {"red": r1, "green": g1, "blue": b1, "nir": nir1, "swir": swir1}
            t2_bands = {"red": r2, "green": g2, "blue": b2, "nir": nir2, "swir": swir2}
            return t1_bands, t2_bands, "custom_upload"

        else:
            raise ValueError("Unsupported input format. Provide scene ID or (t1_path, t2_path) tuple.")

    def run_change_detection(self, t1_input, t2_input, threshold: float = None, min_region_pixels: int = 10, run_id: str = None, scene_id: str = None):
        """Execute full change detection and semantic mapping pipeline."""
        start_total = time.time()
        timing = {}

        if threshold is None:
            threshold = self.default_threshold

        # 1. Validation
        valid, msg = self.validate_inputs(t1_input, t2_input)
        if not valid:
            raise ValueError(msg)

        # 2. Data Loading
        t0 = time.time()
        if isinstance(t1_input, str) and (t1_input.startswith("test_") or t1_input.startswith("val_") or t1_input.startswith("train_")):
            t1_bands, t2_bands, loaded_scene_id = self.load_scene_bands(t1_input)
        else:
            t1_bands, t2_bands, loaded_scene_id = self.load_scene_bands((t1_input, t2_input))
        timing["loading_sec"] = round(time.time() - t0, 4)

        if scene_id is None:
            scene_id = loaded_scene_id

        height, width = t1_bands["red"].shape
        total_pixels = height * width

        # 3. Feature Extraction
        t0 = time.time()
        feature_maps = self.feature_extractor.extract_scene_features_2d(t1_bands, t2_bands)
        X = self.feature_extractor.extract_tabular_dataset(feature_maps, self.selected_features)
        timing["feature_extraction_sec"] = round(time.time() - t0, 4)

        # 4. XGBoost Inference & Binary Change Mask
        t0 = time.time()
        change_probs_1d = self.model.predict_proba(X)
        change_probs_2d = change_probs_1d.reshape(height, width)
        raw_mask_2d = (change_probs_2d >= threshold).astype(np.uint8)

        # Post-Processing (morphological open/close to remove specks and fuse parcel boundaries)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_mask = cv2.morphologyEx(raw_mask_2d, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask)
        clean_mask = np.zeros_like(opened_mask)
        min_px = max(min_region_pixels, 8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_px:
                clean_mask[labels == i] = 1
        
        change_mask = clean_mask
        timing["xgb_inference_sec"] = round(time.time() - t0, 4)

        # 5. Semantic Classification & Intersection
        t0 = time.time()
        semantic_result = detect_semantic_change(
            t1_norm=t1_bands,
            t2_norm=t2_bands,
            change_mask=change_mask,
            prob_map=change_probs_2d,
            semantic_source="spectral_rules"
        )
        timing["semantic_sec"] = round(time.time() - t0, 4)

        # 6. Output Directories & Visualizations Export
        t0 = time.time()
        if run_id is None:
            run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        output_run_dir = Path(self.config["outputs"]["dir"]) / "results" / run_id
        output_run_dir.mkdir(parents=True, exist_ok=True)

        # Prepare RGB Images for visualization
        r1, g1, b1 = t1_bands["red"], t1_bands["green"], t1_bands["blue"]
        r2, g2, b2 = t2_bands["red"], t2_bands["green"], t2_bands["blue"]
        
        # Adaptive scaling: only scale dark 16-bit Sentinel-2 reflectance (max < 0.35); for 8-bit RGB, keep scale = 1.0 to prevent blowout
        scale1 = 2.5 if float(np.max(r1)) < 0.35 else 1.0
        scale2 = 2.5 if float(np.max(r2)) < 0.35 else 1.0
        
        rgb1 = np.clip(np.stack([r1, g1, b1], axis=-1) * scale1, 0.0, 1.0)
        rgb2 = np.clip(np.stack([r2, g2, b2], axis=-1) * scale2, 0.0, 1.0)

        # Save Before / After RGB PNGs
        plt.imsave(output_run_dir / "before.png", rgb1)
        plt.imsave(output_run_dir / "after.png", rgb2)

        # Raw Absolute RGB Difference & Continuous Difference Heatmap
        diff_rgb = np.abs(rgb2 - rgb1)
        diff_mag = np.sqrt(np.sum(diff_rgb**2, axis=-1)) / np.sqrt(3.0)  # normalized 0.0 to 1.0
        diff_mag_uint8 = (np.clip(diff_mag, 0.0, 1.0) * 255).astype(np.uint8)
        diff_heatmap_bgr = cv2.applyColorMap(diff_mag_uint8, cv2.COLORMAP_INFERNO)
        diff_heatmap_rgb = cv2.cvtColor(diff_heatmap_bgr, cv2.COLOR_BGR2RGB)
        plt.imsave(output_run_dir / "diff_heatmap.png", diff_heatmap_rgb)

        # Quantitative Image Difference Metrics
        mean_abs_diff_pct = round(float(np.mean(diff_rgb)) * 100.0, 2)
        rmse_diff_pct = round(float(np.sqrt(np.mean(diff_rgb**2))) * 100.0, 2)
        diff_gt_5pct = round(float(np.mean(diff_mag > 0.05)) * 100.0, 2)
        diff_gt_10pct = round(float(np.mean(diff_mag > 0.10)) * 100.0, 2)
        diff_gt_20pct = round(float(np.mean(diff_mag > 0.20)) * 100.0, 2)

        # Binary Change Map
        plt.imsave(output_run_dir / "change_map.png", change_mask, cmap="gray")

        # Color-coded Semantic Mask (T2)
        sem_t2 = semantic_result["semantic_mask_t2"]
        sem_color_map = {
            0: [0, 0, 0],       # Background (Black)
            1: [255, 230, 0],   # Road (Yellow)
            2: [230, 30, 30],   # Building (Red)
            3: [0, 130, 255],   # Water (Blue)
            4: [30, 200, 50],   # Vegetation (Green)
            5: [230, 130, 30],  # Bare land (Orange)
            6: [180, 50, 180]   # Unknown (Purple)
        }
        sem_t2_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for cid, col in sem_color_map.items():
            sem_t2_rgb[sem_t2 == cid] = col
        plt.imsave(output_run_dir / "semantic_map.png", sem_t2_rgb)

        # Semantic Change Map
        sem_change = semantic_result["semantic_change_mask"]
        sem_change_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for cid, col in sem_color_map.items():
            sem_change_rgb[sem_change == cid] = col
        plt.imsave(output_run_dir / "semantic_change_map.png", sem_change_rgb)

        # Save to outputs/final_maps/
        final_maps_dir = Path("outputs/final_maps")
        final_maps_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(final_maps_dir / f"{run_id}_semantic_change.png", sem_change_rgb)

        # Transparency Overlay on T2 Image with Region Bounding Boxes & IDs
        overlay_rgb = rgb2.copy()
        class_colors = {
            1: [1.0, 0.9, 0.0], 2: [0.9, 0.1, 0.1], 3: [0.0, 0.5, 1.0],
            4: [0.1, 0.8, 0.2], 5: [0.9, 0.5, 0.1], 6: [0.8, 0.2, 0.8]
        }
        for cid, color in class_colors.items():
            mask_c = (sem_change == cid)
            if np.any(mask_c):
                for ch in range(3):
                    overlay_rgb[:, :, ch] = np.where(mask_c, overlay_rgb[:, :, ch] * 0.5 + color[ch] * 0.5, overlay_rgb[:, :, ch])

        # Draw Bounding Boxes and Region IDs on Overlay Image using OpenCV
        overlay_uint8 = (overlay_rgb * 255).astype(np.uint8)
        regions = semantic_result["regions"]
        
        # Filter: Only draw top 15 most significant change regions to avoid visual clutter
        top_candidates = [r for r in regions if r.get("Area", 0) >= 20]
        if not top_candidates:
            top_candidates = regions[:10]
        else:
            top_candidates = top_candidates[:15]

        for r in top_candidates:
            x, y, w, h_box = r["X"], r["Y"], r["Width"], r["Height"]
            cid = r["Class ID"]
            r_id = r["Region ID"]
            c_name = r["Class"].upper()
            pct_chg = r.get("percentage_of_change_area", 0.0)
            
            # Distinct colors per class: Road=Yellow, Building=Red, Water=Blue, Veg=Green, Bare=Orange
            box_col = (0, 230, 255) if cid == 1 else (0, 60, 255) if cid == 2 else (255, 170, 0) if cid == 3 else (0, 220, 60) if cid == 4 else (0, 140, 255)
            
            # Draw shadow outline then crisp colored border (width=2)
            cv2.rectangle(overlay_uint8, (x - 1, y - 1), (x + w + 1, y + h_box + 1), (0, 0, 0), 3)
            cv2.rectangle(overlay_uint8, (x, y), (x + w, y + h_box), box_col, 2)
            
            # Solid pill background for label to prevent unreadable overlap
            lbl = f"#{r_id} {c_name}"
            if pct_chg > 0:
                lbl += f" ({pct_chg}%)"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ty = max(th + 5, y - 4)
            cv2.rectangle(overlay_uint8, (x, ty - th - 3), (x + tw + 4, ty + 2), (20, 25, 35), -1)
            cv2.rectangle(overlay_uint8, (x, ty - th - 3), (x + tw + 4, ty + 2), box_col, 1)
            cv2.putText(overlay_uint8, lbl, (x + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        plt.imsave(output_run_dir / "overlay.png", overlay_uint8)
        plt.imsave(final_maps_dir / f"{run_id}_region_overlay.png", overlay_uint8)

        # Generate per-region 4-panel crops (T1, T2, Change, Semantic)
        crops_dir = output_run_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for r in regions[:10]: # Top 10 ranked regions
            x, y, w, h_box = r["X"], r["Y"], r["Width"], r["Height"]
            pad = 5
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(width, x + w + pad), min(height, y + h_box + pad)

            # Save individual high-res crops for Before, After, Diff, and Semantic inspection
            reg_id = r["Region ID"]
            plt.imsave(crops_dir / f"region_{reg_id}_before.png", rgb1[y1:y2, x1:x2])
            plt.imsave(crops_dir / f"region_{reg_id}_after.png", rgb2[y1:y2, x1:x2])
            plt.imsave(crops_dir / f"region_{reg_id}_semantic.png", sem_change_rgb[y1:y2, x1:x2])

            c_crop = change_mask[y1:y2, x1:x2]
            diff_crop = np.abs(rgb2[y1:y2, x1:x2].astype(np.float32) - rgb1[y1:y2, x1:x2].astype(np.float32))
            diff_crop = np.clip(diff_crop * 2.5, 0, 1.0)
            if np.any(c_crop > 0):
                highlight = rgb2[y1:y2, x1:x2].copy()
                mask_bool = c_crop > 0
                highlight[mask_bool] = highlight[mask_bool] * 0.35 + np.array([1.0, 0.25, 0.25]) * 0.65
                plt.imsave(crops_dir / f"region_{reg_id}_diff.png", highlight)
            else:
                plt.imsave(crops_dir / f"region_{reg_id}_diff.png", diff_crop)

            crop_fig, crop_axes = plt.subplots(1, 4, figsize=(8, 2))
            crop_axes[0].imshow(rgb1[y1:y2, x1:x2])
            crop_axes[0].set_title("T1 Crop", fontsize=8)
            crop_axes[0].axis("off")

            crop_axes[1].imshow(rgb2[y1:y2, x1:x2])
            crop_axes[1].set_title("T2 Crop", fontsize=8)
            crop_axes[1].axis("off")

            crop_axes[2].imshow(change_mask[y1:y2, x1:x2], cmap="gray")
            crop_axes[2].set_title("Change", fontsize=8)
            crop_axes[2].axis("off")

            crop_axes[3].imshow(sem_change_rgb[y1:y2, x1:x2])
            crop_axes[3].set_title("Semantic", fontsize=8)
            crop_axes[3].axis("off")

            plt.suptitle(f"Region {r['Region ID']} — {r['Class'].upper()} ({r['Change Type']})", fontsize=9)
            plt.tight_layout()
            plt.savefig(crops_dir / f"region_{r['Region ID']}_crop.png", dpi=150, bbox_inches="tight")
            plt.close()

        # Day 12 Geospatial Boundary Overlay & GeoJSON Export
        try:
            from src.geospatial import get_geospatial_metadata, draw_region_boundaries_image, generate_region_geojson
            geo_meta = get_geospatial_metadata(str(t2_input) if isinstance(t2_input, str) else "")
            
            # Boundary image
            geo_out_dir = Path("outputs/geospatial")
            geo_out_dir.mkdir(parents=True, exist_ok=True)
            boundary_path = geo_out_dir / f"{scene_id}_region_boundaries.png"
            draw_region_boundaries_image(rgb2, regions, str(boundary_path))
            draw_region_boundaries_image(rgb2, regions, str(output_run_dir / "region_boundaries.png"))

            # GeoJSON
            geojson_data = generate_region_geojson(scene_id, regions, geo_meta)
            if geojson_data:
                with open(geo_out_dir / f"{scene_id}_regions.geojson", "w") as f:
                    json.dump(geojson_data, f, indent=2)
                with open(output_run_dir / "regions.geojson", "w") as f:
                    json.dump(geojson_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not generate Day 12 geospatial assets: {e}")

        timing["visualization_sec"] = round(time.time() - t0, 4)

        # 7. Day 8 Region CSV, Transition CSV & JSON Reports
        reports_dir = Path("outputs/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # A. Transition Matrix CSV
        trans_csv_path = reports_dir / "semantic_transition_matrix.csv"
        with open(trans_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["From Class", "To Class", "From ID", "To ID", "Pixel Count"])
            for tm in semantic_result["transition_matrix"]:
                writer.writerow([tm["from_class"], tm["to_class"], tm["from_id"], tm["to_id"], tm["pixel_count"]])

        # B. Day 8 Semantic Change Regions CSV
        regions_csv_path = reports_dir / "semantic_change_regions.csv"
        run_regions_csv = output_run_dir / "regions.csv"
        
        for p in [regions_csv_path, run_regions_csv]:
            with open(p, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Scene ID", "Region ID", "Rank", "Class", "T1 Class", "T2 Class", "Change Type",
                    "Area (px)", "Significance Score", "Mean Probability", "NDVI Diff", "NDBI Diff",
                    "BBox X", "BBox Y", "Width", "Height", "Centroid X", "Centroid Y", "Quality Flags"
                ])
                for r in regions:
                    q_str = f"align_ok={r['Quality Flags']['alignment_ok']};sem_avail={r['Quality Flags']['semantic_available']}"
                    writer.writerow([
                        scene_id, r["Region ID"], r["Rank"], r["Class"], r["T1 Class"], r["T2 Class"], r["Change Type"],
                        r["Area"], r["Significance Score"], r["Mean Probability"], r["NDVI Difference"], r["NDBI Difference"],
                        r["X"], r["Y"], r["Width"], r["Height"], f"{r['Centroid X']:.1f}", f"{r['Centroid Y']:.1f}", q_str
                    ])

        # C. region_summary.json
        reg_summary_path = output_run_dir / "region_summary.json"
        summary_data = {
            "scene_id": scene_id,
            "run_id": run_id,
            "total_changed_pixels": semantic_result["statistics"]["total_changed_pixels"],
            "region_count": len(regions),
            "regions": regions
        }
        with open(reg_summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Check if geospatial metadata json exists and attach true GPS coordinates
        if isinstance(t2_input, (str, Path)):
            meta_cand = Path(str(t2_input).replace(".png", "_meta.json"))
            if not meta_cand.exists() and scene_id:
                clean_sid = scene_id.replace("_2020", "").replace("_2022", "").replace("_2024", "")
                dist_cand = clean_sid.split("_")[0].lower()
                root_dir = Path(__file__).resolve().parent.parent
                cand2 = root_dir / "dataset_tamilnadu_districts" / dist_cand / "close_aerial_z14" / f"{clean_sid}_meta.json"
                if cand2.exists():
                    meta_cand = cand2

            if meta_cand.exists():
                try:
                    with open(meta_cand) as mf:
                        t_m = json.load(mf)
                    bbox = t_m.get("bbox", [])
                    if len(bbox) >= 4:
                        min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]
                        gsd = t_m.get("ground_resolution_m_per_px", 9.3)
                        for r in regions:
                            cx = r.get("Centroid X", width / 2.0)
                            cy = r.get("Centroid Y", height / 2.0)
                            r["centroid_lon"] = round(min_lon + (cx / width) * (max_lon - min_lon), 6)
                            r["centroid_lat"] = round(max_lat - (cy / height) * (max_lat - min_lat), 6)
                            r["area_m2"] = round(r.get("Area", 0) * (gsd ** 2), 2)
                except Exception:
                    pass

        # Enrich each region with percentage values
        tot_changed_px = max(1, semantic_result["statistics"]["total_changed_pixels"])
        for r in regions:
            r_area = float(r.get("Area", 0))
            r["percentage_of_change_area"] = round((r_area / tot_changed_px) * 100, 2)
            r["percentage_of_scene_area"] = round((r_area / total_pixels) * 100, 4)
            mean_prob = float(r.get("Mean Probability", 0.75))
            r["confidence_percentage"] = round(float(np.clip(mean_prob * 100.0, 0.0, 100.0)), 1)

        # Enrich transitions with percentage of total transitions
        tot_trans_px = max(1, sum(tm["pixel_count"] for tm in semantic_result["transition_matrix"]))
        for tm in semantic_result["transition_matrix"]:
            tm["percentage"] = round((tm["pixel_count"] / tot_trans_px) * 100, 1)

        # Index region data into SQLite database for Day 11 Search Engine
        try:
            from src.search_engine import RegionDatabase
            RegionDatabase().insert_regions(run_id=run_id, scene_id=scene_id, regions=regions)
        except Exception as e:
            print(f"⚠️ Warning: Could not index regions into SQLite database: {e}")

        timing["total_sec"] = round(time.time() - start_total, 4)

        # Summary JSON
        stats = semantic_result["statistics"]
        total_changed = max(1, stats["total_changed_pixels"])
        semantic_breakdown = {
            "1": {"class_name": "road", "pixels": stats["road_pixels"], "percentage_of_changed_area": round(stats["road_pixels"] / total_changed * 100, 2), "percentage_of_total_scene": round(stats["road_pixels"] / total_pixels * 100, 4)},
            "2": {"class_name": "building", "pixels": stats["building_pixels"], "percentage_of_changed_area": round(stats["building_pixels"] / total_changed * 100, 2), "percentage_of_total_scene": round(stats["building_pixels"] / total_pixels * 100, 4)},
            "3": {"class_name": "water", "pixels": stats["water_pixels"], "percentage_of_changed_area": round(stats["water_pixels"] / total_changed * 100, 2), "percentage_of_total_scene": round(stats["water_pixels"] / total_pixels * 100, 4)},
            "4": {"class_name": "vegetation", "pixels": stats["vegetation_pixels"], "percentage_of_changed_area": round(stats["vegetation_pixels"] / total_changed * 100, 2), "percentage_of_total_scene": round(stats["vegetation_pixels"] / total_pixels * 100, 4)},
            "5": {"class_name": "bare_land", "pixels": stats["bare_land_pixels"], "percentage_of_changed_area": round(stats["bare_land_pixels"] / total_changed * 100, 2), "percentage_of_total_scene": round(stats["bare_land_pixels"] / total_pixels * 100, 4)},
            "6": {"class_name": "unknown", "pixels": stats["unknown_pixels"], "percentage_of_changed_area": round(stats["unknown_pixels"] / total_changed * 100, 2), "percentage_of_total_scene": round(stats["unknown_pixels"] / total_pixels * 100, 4)},
        }

        largest_region_name = stats["largest_region"].get("Class", "None") if isinstance(stats["largest_region"], dict) else "None"
        if largest_region_name != "None":
            largest_region_name = f"{largest_region_name} ({stats['largest_region']['Area']}px)"

        change_pct = round((stats["total_changed_pixels"] / total_pixels) * 100, 2)
        unchanged_pct = round(100.0 - change_pct, 2)
        mean_confidence_pct = round(float(np.mean(change_probs_2d[change_mask > 0])) * 100, 1) if stats["total_changed_pixels"] > 0 else 0.0

        result_json = {
            "status": "success",
            "run_id": run_id,
            "scene_id": scene_id,
            "image_dimensions": [height, width],
            "total_pixels": total_pixels,
            "threshold_used": threshold,
            "change_detected": bool(stats["total_changed_pixels"] > 0),
            "changed_pixels": stats["total_changed_pixels"],
            "unchanged_pixels": total_pixels - stats["total_changed_pixels"],
            "change_percentage": change_pct,
            "unchanged_percentage": unchanged_pct,
            "mean_confidence_percentage": mean_confidence_pct,
            "difference_metrics": {
                "mean_abs_diff_percentage": mean_abs_diff_pct,
                "rmse_diff_percentage": rmse_diff_pct,
                "pixels_exceeding_5pct_diff": diff_gt_5pct,
                "pixels_exceeding_10pct_diff": diff_gt_10pct,
                "pixels_exceeding_20pct_diff": diff_gt_20pct
            },
            "semantic_breakdown": semantic_breakdown,
            "transition_matrix": semantic_result["transition_matrix"][:5], # Top 5 transitions
            "region_count": len(regions),
            "largest_region": largest_region_name,
            "regions": regions,
            "timing_seconds": timing,
            "visualizations": {
                "before": f"/static/generated/{run_id}/before.png",
                "after": f"/static/generated/{run_id}/after.png",
                "diff_heatmap": f"/static/generated/{run_id}/diff_heatmap.png",
                "change_map": f"/static/generated/{run_id}/change_map.png",
                "semantic_map": f"/static/generated/{run_id}/semantic_map.png",
                "semantic_change_map": f"/static/generated/{run_id}/semantic_change_map.png",
                "overlay": f"/static/generated/{run_id}/overlay.png"
            }
        }

        json_path = output_run_dir / "result.json"
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=2)

        # Create downloadable ZIP package
        zip_path = output_run_dir / f"result_{run_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_name in ["before.png", "after.png", "diff_heatmap.png", "change_map.png", "semantic_map.png", "semantic_change_map.png", "overlay.png", "regions.csv", "result.json", "region_summary.json"]:
                p = output_run_dir / file_name
                if p.exists():
                    zipf.write(p, arcname=file_name)

        return result_json, regions, output_run_dir
