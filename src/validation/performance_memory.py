import sys
import csv
import resource
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference import InferenceEngine

def get_mem_mb():
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def run_performance_memory_benchmark():
    print("=" * 80)
    print("⚡ DAY 6 PERFORMANCE & MEMORY BENCHMARK (3 RUNS PER SCENE)")
    print("=" * 80)

    engine = InferenceEngine(config_path="config.yaml")

    scenes = ["test_000", "test_001", "test_002"]
    num_runs = 3

    metrics_store = {
        "loading_sec": [],
        "feature_extraction_sec": [],
        "xgb_inference_sec": [],
        "semantic_sec": [],
        "visualization_sec": [],
        "total_sec": []
    }

    initial_mem_mb = get_mem_mb()
    print(f"Initial Memory Usage: {initial_mem_mb:.2f} MB")

    for scene in scenes:
        for run_idx in range(1, num_runs + 1):
            mem_before = get_mem_mb()
            res_json, regions, _ = engine.run_change_detection(
                t1_input=scene,
                t2_input=scene,
                run_id=f"perf_{scene}_run{run_idx}"
            )
            mem_after = get_mem_mb()
            t = res_json["timing_seconds"]

            for k in metrics_store.keys():
                metrics_store[k].append(t[k])

            print(f"Scene {scene} (Run {run_idx}): Total = {t['total_sec']:.4f}s | XGBoost = {t['xgb_inference_sec']:.4f}s | Mem = {mem_after:.2f} MB (Delta: +{mem_after - mem_before:.2f} MB)")

    # Compute Summary Stats
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    perf_csv_path = reports_dir / "day6_performance.csv"

    with open(perf_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Component", "Average Time (s)", "Min Time (s)", "Max Time (s)"])

        print("\n" + "-" * 70)
        print(f"{'Component':<25} | {'Average (s)':<12} | {'Min (s)':<10} | {'Max (s)':<10}")
        print("-" * 70)

        for comp, vals in metrics_store.items():
            avg_v = sum(vals) / len(vals)
            min_v = min(vals)
            max_v = max(vals)
            writer.writerow([comp, f"{avg_v:.4f}", f"{min_v:.4f}", f"{max_v:.4f}"])
            print(f"{comp:<25} | {avg_v:<12.4f} | {min_v:<10.4f} | {max_v:<10.4f}")

    peak_mem_mb = get_mem_mb()
    print("-" * 70)
    print(f"Peak RSS Memory Usage: {peak_mem_mb:.2f} MB")
    print(f"📊 Day 6 Performance Report saved to: {perf_csv_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_performance_memory_benchmark()
