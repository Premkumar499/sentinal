import numpy as np
import cv2
import yaml
from src.features.extraction import calculate_ndvi, calculate_ndbi

class FeatureExtractorDay3:
    """Advanced feature extractor for Day 3 including temporal differences, relative changes, and spatial context."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.band_keys = ["red", "green", "blue", "nir", "swir"]

    def extract_scene_features_2d(self, t1_bands: dict, t2_bands: dict) -> dict:
        """
        Extract 2D feature maps for a scene.
        Input dicts must contain keys: 'red', 'green', 'blue', 'nir', 'swir'
        """
        r1, g1, b1 = t1_bands["red"], t1_bands["green"], t1_bands["blue"]
        r2, g2, b2 = t2_bands["red"], t2_bands["green"], t2_bands["blue"]
        nir1, nir2 = t1_bands["nir"], t2_bands["nir"]
        swir1, swir2 = t1_bands["swir"], t2_bands["swir"]

        # 1. Base 16 features
        r_diff = r2 - r1
        g_diff = g2 - g1
        b_diff = b2 - b1
        rgb_diff_mag = np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)

        ndvi1 = calculate_ndvi(nir1, r1)
        ndvi2 = calculate_ndvi(nir2, r2)
        ndvi_diff = ndvi2 - ndvi1

        ndbi1 = calculate_ndbi(swir1, nir1)
        ndbi2 = calculate_ndbi(swir2, nir2)
        ndbi_diff = ndbi2 - ndbi1

        features = {
            "R_T1": r1, "G_T1": g1, "B_T1": b1,
            "R_T2": r2, "G_T2": g2, "B_T2": b2,
            "R_diff": r_diff, "G_diff": g_diff, "B_diff": b_diff,
            "NDVI_T1": ndvi1, "NDVI_T2": ndvi2, "NDVI_diff": ndvi_diff,
            "NDBI_T1": ndbi1, "NDBI_T2": ndbi2, "NDBI_diff": ndbi_diff,
            "RGB_diff_mag": rgb_diff_mag
        }

        # 2. Improved Temporal Difference Features (Section 11)
        features["R_abs_diff"] = np.abs(r_diff)
        features["G_abs_diff"] = np.abs(g_diff)
        features["B_abs_diff"] = np.abs(b_diff)
        features["NDVI_abs_diff"] = np.abs(ndvi_diff)
        features["NDBI_abs_diff"] = np.abs(ndbi_diff)

        # Relative change where stable (epsilon = 1e-4)
        features["R_rel_change"] = r_diff / (np.abs(r1) + 1e-4)
        features["G_rel_change"] = g_diff / (np.abs(g1) + 1e-4)
        features["B_rel_change"] = b_diff / (np.abs(b1) + 1e-4)
        features["NDVI_rel_change"] = ndvi_diff / (np.abs(ndvi1) + 1e-4)

        # 3. CPU-friendly Spatial Features (Section 12)
        # Local 3x3 mean using OpenCV blur (very fast)
        features["mean_3x3_T1_red"] = cv2.blur(r1, (3, 3))
        features["mean_3x3_T2_red"] = cv2.blur(r2, (3, 3))
        features["mean_3x3_T1_nir"] = cv2.blur(nir1, (3, 3))
        features["mean_3x3_T2_nir"] = cv2.blur(nir2, (3, 3))
        
        # Neighborhood mean difference
        features["mean_3x3_RGB_diff_mag"] = cv2.blur(rgb_diff_mag, (3, 3))
        features["delta_mean_3x3_red"] = features["mean_3x3_T2_red"] - features["mean_3x3_T1_red"]

        # Local standard deviation of RGB diff mag
        mean_sq = cv2.blur(rgb_diff_mag**2, (3, 3))
        sq_mean = features["mean_3x3_RGB_diff_mag"]**2
        features["std_3x3_RGB_diff_mag"] = np.sqrt(np.maximum(0, mean_sq - sq_mean))

        return features

    def extract_tabular_dataset(self, feature_maps: dict, feature_names_list: list, ground_truth: np.ndarray = None):
        """Flatten 2D feature maps into tabular X matrix (N_pixels, N_features) and y label vector."""
        height, width = feature_maps["R_T1"].shape
        n_pixels = height * width
        
        feature_cols = []
        for col_name in feature_names_list:
            feature_cols.append(feature_maps[col_name].reshape(n_pixels, 1))

        X = np.hstack(feature_cols).astype(np.float32)

        if ground_truth is not None:
            y = ground_truth.reshape(n_pixels).astype(np.int32)
            return X, y
        return X
