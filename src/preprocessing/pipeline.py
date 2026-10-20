import os
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import yaml

class OSCDLoader:
    """Loader for the OSCD dataset scenes, bands, and ground-truth masks."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config["dataset"]["raw_dir"])
        self.images_dir = self.raw_dir / self.config["dataset"]["images_subdir"]
        self.train_labels_dir = self.raw_dir / self.config["dataset"]["train_labels_subdir"]
        self.test_labels_dir = self.raw_dir / self.config["dataset"]["test_labels_subdir"]
        self.val_labels_dir = self.raw_dir / self.config["dataset"]["val_labels_subdir"]
        self.band_mapping = self.config["band_mapping"]

    def list_scenes(self, split: str = "train"):
        """Return list of available scene IDs for a given split (train, test, val)."""
        label_dir = getattr(self, f"{split}_labels_dir")
        if not label_dir.exists():
            return []
        return sorted([d.name for d in label_dir.iterdir() if d.is_dir()])

    def get_scene_paths(self, scene_id: str):
        """Locate image directories and label mask for a specific scene."""
        scene_img_dir = self.images_dir / scene_id
        if not scene_img_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_img_dir}")
        
        t1_dir = scene_img_dir / "imgs_1_rect"
        t2_dir = scene_img_dir / "imgs_2_rect"

        # Determine label directory
        label_path = None
        for ldir in [self.train_labels_dir, self.test_labels_dir, self.val_labels_dir]:
            candidate = ldir / scene_id / "cm" / "cm.png"
            if candidate.exists():
                label_path = candidate
                break

        return {
            "t1_dir": t1_dir,
            "t2_dir": t2_dir,
            "label_path": label_path
        }

    def load_band(self, band_file: Path) -> np.ndarray:
        """Load a single raster band using rasterio."""
        if not band_file.exists():
            raise FileNotFoundError(f"Band file not found: {band_file}")
        with rasterio.open(band_file) as src:
            data = src.read(1).astype(np.float32)
        return data

    def load_ground_truth(self, label_path: Path) -> np.ndarray:
        """Load and binarize the ground-truth change mask (0=unchanged, 1=changed)."""
        if label_path is None or not label_path.exists():
            raise FileNotFoundError(f"Ground truth label path does not exist: {label_path}")
        mask = plt.imread(label_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Binarize threshold
        binary_mask = (mask > 0.5).astype(np.int32)
        return binary_mask


class PreprocessingPipeline:
    """Preprocessing layer: spatial alignment verification, NaN handling, and reflectance normalization."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.scale_factor = float(self.config["preprocessing"]["reflectance_scale"])
        self.nan_fill_value = float(self.config["preprocessing"]["nan_fill_value"])

    def verify_and_align(self, t1_dict: dict, t2_dict: dict, ground_truth: np.ndarray):
        """Verify spatial dimensions across T1, T2 bands, and ground-truth mask."""
        ref_shape = ground_truth.shape
        for key, arr in t1_dict.items():
            if arr.shape != ref_shape:
                raise ValueError(f"T1 Band {key} shape {arr.shape} mismatch with GT mask shape {ref_shape}")
        for key, arr in t2_dict.items():
            if arr.shape != ref_shape:
                raise ValueError(f"T2 Band {key} shape {arr.shape} mismatch with GT mask shape {ref_shape}")

    def clean_and_normalize(self, band_data: np.ndarray) -> np.ndarray:
        """Handle NaNs and scale digital numbers to normalized reflectance values [0.0, 1.0]."""
        # Fill NaN / Inf
        cleaned = np.nan_to_num(band_data, nan=self.nan_fill_value, posinf=self.nan_fill_value, neginf=self.nan_fill_value)
        # Normalize to reflectance
        normalized = cleaned / self.scale_factor
        return np.clip(normalized, 0.0, 1.0)
