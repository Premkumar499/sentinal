import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def prepare_rgb_display(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Prepare 3-channel RGB image normalized with 2nd/98th percentiles for visual clarity."""
    rgb = np.stack([r, g, b], axis=-1)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb_norm = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        rgb_norm = np.clip(rgb, 0.0, 1.0)
    return rgb_norm

def create_day1_visualization(
    t1_bands: dict,
    t2_bands: dict,
    ground_truth: np.ndarray,
    simple_diff_map: np.ndarray,
    scene_id: str,
    save_path: str
):
    """
    Generate and save the 4-panel Day 1 milestone visualization:
    1. BEFORE (T1 RGB)
    2. AFTER (T2 RGB)
    3. GROUND TRUTH (Actual binary change mask)
    4. SIMPLE CHANGE (Detected Euclidean difference baseline map)
    """
    t1_rgb = prepare_rgb_display(t1_bands["red"], t1_bands["green"], t1_bands["blue"])
    t2_rgb = prepare_rgb_display(t2_bands["red"], t2_bands["green"], t2_bands["blue"])

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=150)
    fig.suptitle(f"OSCD Change Detection Day 1 Pipeline — Scene: {scene_id}", fontsize=16, fontweight='bold', y=0.98)

    # 1. BEFORE
    axes[0, 0].imshow(t1_rgb)
    axes[0, 0].set_title("BEFORE (T1 RGB Composite)", fontsize=13, fontweight='bold', pad=10, color="#4FC3F7")
    axes[0, 0].axis("off")

    # 2. AFTER
    axes[0, 1].imshow(t2_rgb)
    axes[0, 1].set_title("AFTER (T2 RGB Composite)", fontsize=13, fontweight='bold', pad=10, color="#81C784")
    axes[0, 1].axis("off")

    # 3. GROUND TRUTH
    im_gt = axes[1, 0].imshow(ground_truth, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("GROUND TRUTH (OSCD Mask)", fontsize=13, fontweight='bold', pad=10, color="#FFB74D")
    axes[1, 0].axis("off")
    cbar_gt = fig.colorbar(im_gt, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar_gt.set_ticks([0, 1])
    cbar_gt.set_ticklabels(['Unchanged', 'Changed'])

    # 4. SIMPLE CHANGE
    im_diff = axes[1, 1].imshow(simple_diff_map, cmap="plasma")
    axes[1, 1].set_title("SIMPLE DIFFERENCE (Baseline Map)", fontsize=13, fontweight='bold', pad=10, color="#E57373")
    axes[1, 1].axis("off")
    cbar_diff = fig.colorbar(im_diff, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar_diff.set_label('Difference Magnitude', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_p, bbox_inches="tight", dpi=150)
    plt.close()
    return str(save_p)


def create_day2_visualization(
    t1_bands: dict,
    t2_bands: dict,
    ground_truth: np.ndarray,
    predicted_mask: np.ndarray,
    predicted_prob: np.ndarray,
    scene_id: str,
    save_path: str
) -> str:
    """
    Generate and save a 6-panel Day 2 evaluation dashboard:
    1. BEFORE (T1 RGB)
    2. AFTER (T2 RGB)
    3. GROUND TRUTH (OSCD Mask)
    4. PREDICTED MASK (XGBoost Change Map)
    5. PROBABILITY MAP (Confidence)
    6. ERROR ANALYSIS (TP/TN/FP/FN Breakdown)
    """
    t1_rgb = prepare_rgb_display(t1_bands["red"], t1_bands["green"], t1_bands["blue"])
    t2_rgb = prepare_rgb_display(t2_bands["red"], t2_bands["green"], t2_bands["blue"])

    # Create error category map
    # 0: True Negative (TN) - dark grey
    # 1: True Positive (TP) - green
    # 2: False Positive (FP) - red
    # 3: False Negative (FN) - blue
    error_map = np.zeros_like(ground_truth, dtype=np.int32)
    error_map[(ground_truth == 0) & (predicted_mask == 0)] = 0
    error_map[(ground_truth == 1) & (predicted_mask == 1)] = 1
    error_map[(ground_truth == 0) & (predicted_mask == 1)] = 2
    error_map[(ground_truth == 1) & (predicted_mask == 0)] = 3

    from matplotlib.colors import ListedColormap
    error_colors = ["#222222", "#2ca02c", "#d62728", "#1f77b4"]
    error_cmap = ListedColormap(error_colors)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    fig.suptitle(f"OSCD XGBoost Change Detection Day 2 Evaluation — Scene: {scene_id}", fontsize=18, fontweight='bold', y=0.98)

    # 1. BEFORE
    axes[0, 0].imshow(t1_rgb)
    axes[0, 0].set_title("BEFORE (T1 RGB)", fontsize=13, fontweight='bold', pad=10, color="#4FC3F7")
    axes[0, 0].axis("off")

    # 2. AFTER
    axes[0, 1].imshow(t2_rgb)
    axes[0, 1].set_title("AFTER (T2 RGB)", fontsize=13, fontweight='bold', pad=10, color="#81C784")
    axes[0, 1].axis("off")

    # 3. GROUND TRUTH
    im_gt = axes[0, 2].imshow(ground_truth, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("GROUND TRUTH (OSCD Mask)", fontsize=13, fontweight='bold', pad=10, color="#FFB74D")
    axes[0, 2].axis("off")
    cbar_gt = fig.colorbar(im_gt, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar_gt.set_ticks([0, 1])
    cbar_gt.set_ticklabels(['Unchanged', 'Changed'])

    # 4. PREDICTED MASK
    im_pred = axes[1, 0].imshow(predicted_mask, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("XGBOOST PREDICTED MASK", fontsize=13, fontweight='bold', pad=10, color="#E57373")
    axes[1, 0].axis("off")
    cbar_pred = fig.colorbar(im_pred, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar_pred.set_ticks([0, 1])
    cbar_pred.set_ticklabels(['Unchanged', 'Changed'])

    # 5. PROBABILITY MAP
    im_prob = axes[1, 1].imshow(predicted_prob, cmap="viridis", vmin=0, vmax=1)
    axes[1, 1].set_title("PROBABILITY MAP", fontsize=13, fontweight='bold', pad=10, color="#CE93D8")
    axes[1, 1].axis("off")
    cbar_prob = fig.colorbar(im_prob, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. ERROR ANALYSIS
    im_err = axes[1, 2].imshow(error_map, cmap=error_cmap, vmin=0, vmax=3)
    axes[1, 2].set_title("ERROR ANALYSIS MAP", fontsize=13, fontweight='bold', pad=10, color="#90A4AE")
    axes[1, 2].axis("off")
    cbar_err = fig.colorbar(im_err, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar_err.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar_err.set_ticklabels(['TN', 'TP', 'FP', 'FN'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_p, bbox_inches="tight", dpi=150)
    plt.close()
    return str(save_p)

