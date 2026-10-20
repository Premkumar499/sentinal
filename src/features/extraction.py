import numpy as np
import yaml

def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)
    """
    nir = nir.astype(np.float64)
    red = red.astype(np.float64)
    denominator = nir + red
    return np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir, dtype=np.float64),
        where=denominator != 0
    )


def calculate_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Built-up Index (NDBI).
    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """
    swir = swir.astype(np.float64)
    nir = nir.astype(np.float64)
    denominator = swir + nir
    return np.divide(
        swir - nir,
        denominator,
        out=np.zeros_like(swir, dtype=np.float64),
        where=denominator != 0
    )


class FeatureExtractor:
    """Extract engineered temporal and spectral features for semantic change detection."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.feature_names = self.config["features"]["feature_names"]

    def extract_scene_features_2d(self, t1_bands: dict, t2_bands: dict) -> dict:
        """
        Extract 2D feature maps for a scene.
        Input dicts must contain keys: 'red', 'green', 'blue', 'nir', 'swir'
        """
        r1, g1, b1 = t1_bands["red"], t1_bands["green"], t1_bands["blue"]
        r2, g2, b2 = t2_bands["red"], t2_bands["green"], t2_bands["blue"]
        nir1, nir2 = t1_bands["nir"], t2_bands["nir"]
        swir1, swir2 = t1_bands["swir"], t2_bands["swir"]

        # Temporal RGB differences
        r_diff = r2 - r1
        g_diff = g2 - g1
        b_diff = b2 - b1
        rgb_diff_mag = np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)

        # Spectral Indices
        ndvi1 = calculate_ndvi(nir1, r1)
        ndvi2 = calculate_ndvi(nir2, r2)
        ndvi_diff = ndvi2 - ndvi1

        ndbi1 = calculate_ndbi(swir1, nir1)
        ndbi2 = calculate_ndbi(swir2, nir2)
        ndbi_diff = ndbi2 - ndbi1

        feature_maps = {
            "R_T1": r1,
            "G_T1": g1,
            "B_T1": b1,
            "R_T2": r2,
            "G_T2": g2,
            "B_T2": b2,
            "R_diff": r_diff,
            "G_diff": g_diff,
            "B_diff": b_diff,
            "NDVI_T1": ndvi1,
            "NDVI_T2": ndvi2,
            "NDVI_diff": ndvi_diff,
            "NDBI_T1": ndbi1,
            "NDBI_T2": ndbi2,
            "NDBI_diff": ndbi_diff,
            "RGB_diff_mag": rgb_diff_mag
        }
        return feature_maps

    def extract_tabular_dataset(self, feature_maps: dict, ground_truth: np.ndarray = None):
        """
        Flatten 2D feature maps into tabular X matrix (N_pixels, N_features) and y label vector.
        """
        height, width = feature_maps["R_T1"].shape
        n_pixels = height * width
        
        feature_cols = []
        for col_name in self.feature_names:
            feature_cols.append(feature_maps[col_name].reshape(n_pixels, 1))

        X = np.hstack(feature_cols).astype(np.float32)

        if ground_truth is not None:
            y = ground_truth.reshape(n_pixels).astype(np.int32)
            return X, y
        return X
