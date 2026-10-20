import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import OSCDLoader
from src.inference import InferenceEngine

def run_error_analysis(test_scenes=["test_000", "test_001", "test_002"]):
    print("=" * 70)
    print("📊 RUNNING DAY 6 CHANGE MAP ERROR ANALYSIS (TP/TN/FP/FN)")
    print("=" * 70)

    loader = OSCDLoader(config_path="config.yaml")
    engine = InferenceEngine(config_path="config.yaml")
    out_dir = Path("outputs/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene_id in test_scenes:
        paths = loader.get_scene_paths(scene_id)
        gt_mask = loader.load_ground_truth(paths["label_path"]).astype(np.uint8)

        # Run inference
        res_json, regions, run_dir = engine.run_change_detection(scene_id, scene_id, run_id=f"err_{scene_id}")
        
        # Load raw binary mask
        change_mask_img = (plt.imread(run_dir / "change_map.png")[:, :, 0] > 0.5).astype(np.uint8)

        # Compute confusion maps
        tp_mask = np.logical_and(gt_mask == 1, change_mask_img == 1)
        tn_mask = np.logical_and(gt_mask == 0, change_mask_img == 0)
        fp_mask = np.logical_and(gt_mask == 0, change_mask_img == 1)
        fn_mask = np.logical_and(gt_mask == 1, change_mask_img == 0)

        tp = int(np.sum(tp_mask))
        tn = int(np.sum(tn_mask))
        fp = int(np.sum(fp_mask))
        fn = int(np.sum(fn_mask))

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        iou = tp / max(1, tp + fp + fn)

        print(f"\nScene: {scene_id}")
        print(f"  TP = {tp:,} | TN = {tn:,} | FP = {fp:,} | FN = {fn:,}")
        print(f"  Precision = {prec:.4f} | Recall = {rec:.4f} | F1 = {f1:.4f} | IoU = {iou:.4f}")

        # Create Error Analysis Plot (4 panels: Ground Truth, TP Map, FP Map, FN Map)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(gt_mask, cmap="gray")
        axes[0].set_title(f"{scene_id} Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(tp_mask, cmap="Greens")
        axes[1].set_title(f"True Positives ({tp:,} px)")
        axes[1].axis("off")

        axes[2].imshow(fp_mask, cmap="Reds")
        axes[2].set_title(f"False Positives ({fp:,} px)")
        axes[2].axis("off")

        axes[3].imshow(fn_mask, cmap="Blues")
        axes[3].set_title(f"False Negatives ({fn:,} px)")
        axes[3].axis("off")

        plt.tight_layout()
        plot_path = out_dir / f"{scene_id}_error_analysis.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"  Saved Error Map to: {plot_path}")

    print("\n✅ CHANGE MAP ERROR ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_error_analysis()
