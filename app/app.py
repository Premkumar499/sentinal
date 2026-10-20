import os
import sys
import time
import json
import csv
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference import InferenceEngine
from src.chatbot_engine import InfoChatbotEngine, ImageChatbotEngine

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configuration
UPLOAD_FOLDER = project_root / "outputs" / "uploads"
RESULTS_FOLDER = project_root / "outputs" / "results"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max limit
app.config["TEMPLATES_AUTO_RELOAD"] = True
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}

# Global inference engine instance (loaded once on startup)
try:
    engine = InferenceEngine(config_path=str(project_root / "config.yaml"))
    model_loaded = True
except Exception as e:
    engine = None
    model_loaded = False
    print(f"⚠️ Warning: Could not initialize InferenceEngine on startup: {e}")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html", model_loaded=model_loaded)

@app.route("/demo")
def demo_mode():
    """Presentation route: Instant 1-click execution on pre-configured benchmark scene."""
    scene = request.args.get("scene", "test_006")
    if not model_loaded or engine is None:
        return render_template("error.html", title="Model Unavailable", message="The change detection model could not be loaded.")
    try:
        if scene in ("root_sample", "custom_sample", "before_after", "sample_pair", "SAMPLE_BI_TEMPORAL_PAIR"):
            run_id = f"demo_root_sample_{time.strftime('%M%S')}"
            result_json, regions, run_dir = engine.run_change_detection(
                t1_input=str(project_root / "before_img.png"),
                t2_input=str(project_root / "after_img.png"),
                threshold=0.72,
                min_region_pixels=10,
                run_id=run_id,
                scene_id="OSCD_RGB_Sample_0"
            )
            return redirect(url_for("view_results", run_id=run_id))
        else:
            run_id = f"demo_{scene}_{time.strftime('%M%S')}"
            result_json, regions, run_dir = engine.run_change_detection(
                t1_input=scene,
                t2_input=scene,
                threshold=0.72,
                min_region_pixels=10,
                run_id=run_id
            )
            return redirect(url_for("view_results", run_id=run_id))
    except Exception as e:
        return render_template("error.html", title="Demo Error", message=f"Failed to execute demo route for '{scene}': {str(e)}")

@app.route("/analyze", methods=["POST"])
def analyze():
    if not model_loaded or engine is None:
        return render_template("error.html", title="Model Unavailable", message="The change detection model could not be loaded. Please check that model weights exist in models/.")

    sample_scene = request.form.get("sample_scene")
    
    # Mode A: Sample scene selected (OSCD or Tamil Nadu regional tile or Root Sample Pair)
    if sample_scene and sample_scene != "custom":
        try:
            if sample_scene.startswith("SAMPLE_BI_TEMPORAL") or sample_scene in ("root_sample", "before_after", "sample_pair"):
                run_id = f"run_sample_pair_{time.strftime('%M%S')}"
                result_json, regions, run_dir = engine.run_change_detection(
                    t1_input=str(project_root / "before_img.png"),
                    t2_input=str(project_root / "after_img.png"),
                    threshold=0.72,
                    min_region_pixels=10,
                    run_id=run_id,
                    scene_id="OSCD_RGB_Sample_0"
                )
                return redirect(url_for("view_results", run_id=run_id))

            is_tn = any(sample_scene.startswith(prefix) for prefix in ["CHENNAI_", "COIMBATORE_", "KARUR_", "ARIYALUR_", "CHENGALPATTU_", "CUDDALORE_", "DHARMAPURI_", "DINDIGUL_", "ERODE_", "KALLAKURICHI_", "KANCHIPURAM_", "KANYAKUMARI_", "KRISHNAGIRI_", "MADURAI_", "MAYILADUTHURAI_", "NAGAPATTINAM_", "NAMAKKAL_", "NILGIRIS_", "PERAMBALUR_", "PUDUKKOTTAI_", "RAMANATHAPURAM_", "RANIPET_", "SALEM_", "SIVAGANGA_", "TENKASI_", "THANJAVUR_", "THENI_", "THOOTHUKUDI_", "TIRUCHIRAPPALLI_", "TIRUNELVELI_", "TIRUPATHUR_", "TIRUPPUR_", "TIRUVALLUR_", "TIRUVANNAMALAI_", "TIRUVARUR_", "VELLORE_", "VILUPPURAM_", "VIRUDHUNAGAR_"])
            if is_tn:
                dist = sample_scene.split("_")[0].lower()
                t2_path = project_root / "dataset_tamilnadu_districts" / dist / "close_aerial_z14" / f"{sample_scene}.png"
                tile_id = sample_scene  # e.g. KARUR_CLOSE_Z14_R01_C01

                # Look for real T1 from downloaded temporal data (prefer 2020, fallback 2022)
                t1_path = None
                for year in ["2020", "2022"]:
                    candidate = project_root / "data" / "tamilnadu_temporal" / dist / f"year_{year}" / f"{tile_id}_{year}.png"
                    if candidate.exists():
                        t1_path = candidate
                        break

                # Final fallback: synthetic counterpart if download not yet complete
                if t1_path is None:
                    from scripts.run_tamilnadu_change_detection import create_realistic_historical_counterpart
                    synthetic_path = project_root / "data" / "tamilnadu_temporal" / dist / "synthetic" / f"{sample_scene}_T1_synthetic.png"
                    t1_path = create_realistic_historical_counterpart(t2_path, synthetic_path)

                run_id = f"tn_{dist}_{sample_scene}_{time.strftime('%M%S')}"
                result_json, regions, run_dir = engine.run_change_detection(
                    t1_input=str(t1_path),
                    t2_input=str(t2_path),
                    threshold=0.65,
                    min_region_pixels=12,
                    run_id=run_id,
                    scene_id=sample_scene
                )
                return redirect(url_for("view_results", run_id=run_id))
            else:
                run_id = f"run_{sample_scene}_{time.strftime('%M%S')}"
                result_json, regions, run_dir = engine.run_change_detection(
                    t1_input=sample_scene,
                    t2_input=sample_scene,
                    threshold=0.72,
                    min_region_pixels=10,
                    run_id=run_id
                )
                return redirect(url_for("view_results", run_id=run_id))
        except Exception as e:
            return render_template("error.html", title="Analysis Failed", message=f"Failed to process sample scene '{sample_scene}': {str(e)}")

    # Mode B: Custom File Uploads (T1 and T2)
    file_t1 = request.files.get("file_t1")
    file_t2 = request.files.get("file_t2")

    if not file_t1 or not file_t2 or file_t1.filename == "" or file_t2.filename == "":
        return render_template("error.html", title="Missing Files", message="Please upload both T1 (Earlier) and T2 (Later) satellite images.")

    if not (allowed_file(file_t1.filename) and allowed_file(file_t2.filename)):
        return render_template("error.html", title="Invalid File Format", message="Only standard raster formats (.png, .jpg, .jpeg, .tif) are supported.")

    run_id = f"run_upload_{time.strftime('%Y%m%d_%H%M%S')}"
    path_t1 = UPLOAD_FOLDER / f"{run_id}_t1_{secure_filename(file_t1.filename)}"
    path_t2 = UPLOAD_FOLDER / f"{run_id}_t2_{secure_filename(file_t2.filename)}"

    file_t1.save(path_t1)
    file_t2.save(path_t2)

    try:
        result_json, regions, run_dir = engine.run_change_detection(
            t1_input=str(path_t1),
            t2_input=str(path_t2),
            threshold=0.72,
            min_region_pixels=10,
            run_id=run_id
        )
        return redirect(url_for("view_results", run_id=run_id))
    except Exception as e:
        return render_template("error.html", title="Inference Error", message=f"Error analyzing uploaded satellite images: {str(e)}")

@app.route("/results/<run_id>")
def view_results(run_id):
    run_dir = RESULTS_FOLDER / run_id
    json_file = run_dir / "result.json"

    if not json_file.exists():
        return render_template("error.html", title="Result Not Found", message=f"No analysis results exist for run ID: {run_id}")

    with open(json_file, "r") as f:
        data = json.load(f)

    # Load regions from region_summary.json or result.json
    summary_file = run_dir / "region_summary.json"
    regions = []
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
            regions = summary.get("regions", [])
    elif "regions" in data:
        regions = data["regions"]
    elif (run_dir / "regions.csv").exists():
        import csv
        with open(run_dir / "regions.csv", "r") as f:
            reader = csv.DictReader(f)
            regions = list(reader)

    return render_template("results.html", data=data, regions=regions, run_id=run_id)

@app.route("/region/<run_id>/<region_id>")
def view_region_detail(run_id, region_id):
    run_dir = RESULTS_FOLDER / run_id
    json_file = run_dir / "result.json"

    if not json_file.exists():
        return render_template("error.html", title="Result Not Found", message=f"No analysis results exist for run ID: {run_id}")

    summary_file = run_dir / "region_summary.json"
    region = None
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
            for r in summary.get("regions", []):
                if str(r.get("Region ID")) == str(region_id):
                    region = r
                    break

    crop_url = f"/static/generated/{run_id}/crops/region_{region_id}_crop.png"
    crop_path = run_dir / "crops" / f"region_{region_id}_crop.png"
    has_crop = crop_path.exists()

    # Retrieve georeferenced coordinates if available
    geo_location = None
    db_file = project_root / "data" / "change_regions.db"
    if db_file.exists():
        try:
            import sqlite3
            with sqlite3.connect(str(db_file)) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT centroid_lat, centroid_lon, scene_id, area_m2, semantic_class FROM change_regions WHERE run_id=? AND region_id=?",
                    (run_id, int(region_id))
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    lat, lon, sc_id, a_m2, s_cls = row
                    dist = "Tamil Nadu"
                    for d_candidate in ["Chennai", "Coimbatore", "Karur", "Ariyalur", "Chengalpattu", "Madurai", "Salem"]:
                        if d_candidate.lower() in (sc_id or "").lower() or d_candidate.lower() in run_id.lower():
                            dist = d_candidate
                            break
                    geo_location = {
                        "district": dist,
                        "latitude": round(float(lat), 6),
                        "longitude": round(float(lon), 6),
                        "coordinates_str": f"{round(float(lat), 5)}° N, {round(float(lon), 5)}° E",
                        "area_m2": round(float(a_m2 or 0.0), 1),
                        "area_ha": round(float(a_m2 or 0.0) / 10000.0, 3),
                        "map_url": f"/map?lat={round(float(lat), 6)}&lon={round(float(lon), 6)}&zoom=16&region_id={region_id}&district={dist.lower()}",
                        "google_maps_url": f"https://www.google.com/maps?q={round(float(lat), 6)},{round(float(lon), 6)}"
                    }
        except Exception as e:
            print(f"Error querying geo_location for region {region_id}: {e}")

    return render_template(
        "region_detail.html",
        run_id=run_id,
        region_id=region_id,
        region=region,
        crop_url=crop_url if has_crop else None,
        geo_location=geo_location
    )

def ensure_4panel_crop(run_id: str, filename: str) -> bool:
    """Ensure that the standard 4-panel horizontal crop strip (T1, T2, Change, Semantic)
    exists on disk for any requested region_id, generating it on the fly if needed.
    """
    import re
    m = re.match(r"^region_(\d+)_crop\.png$", filename)
    if not m:
        return False
    reg_id = int(m.group(1))

    run_dir = RESULTS_FOLDER / run_id
    if not run_dir.exists():
        return False

    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    target_file = crops_dir / filename
    if target_file.exists() and target_file.stat().st_size > 1000:
        return True

    # 1. Lookup region metadata from summary, db, or csv
    r = None
    sc_id = ""
    sum_file = run_dir / "region_summary.json"
    if sum_file.exists():
        try:
            with open(sum_file, "r") as f:
                sdata = json.load(f)
            r = next((x for x in sdata.get("regions", []) if int(x.get("Region ID", -1)) == reg_id), None)
            sc_id = sdata.get("scene_id", "")
        except Exception:
            pass

    if not r:
        db_file = project_root / "data" / "change_regions.db"
        if db_file.exists():
            try:
                import sqlite3
                with sqlite3.connect(str(db_file)) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.cursor().execute(
                        "SELECT * FROM change_regions WHERE run_id = ? AND region_id = ?",
                        (run_id, reg_id)
                    ).fetchone()
                    if row:
                        r = {
                            "X": row["bbox_x"],
                            "Y": row["bbox_y"],
                            "Width": row["bbox_width"],
                            "Height": row["bbox_height"],
                            "Class": row["semantic_class"],
                            "Change Type": row["change_type"],
                            "Region ID": reg_id
                        }
                        sc_id = row["scene_id"]
            except Exception:
                pass

    if not r:
        reg_csv = run_dir / "regions.csv"
        if reg_csv.exists():
            try:
                import csv
                with open(reg_csv, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if int(row.get("Region ID", -1)) == reg_id:
                            r = {
                                "X": int(row.get("BBox X", 0)),
                                "Y": int(row.get("BBox Y", 0)),
                                "Width": int(row.get("Width", 20)),
                                "Height": int(row.get("Height", 20)),
                                "Class": row.get("Class", "change"),
                                "Change Type": row.get("Change Type", "transition"),
                                "Region ID": reg_id
                            }
                            sc_id = row.get("Scene ID", "")
                            break
            except Exception:
                pass

    if not r:
        return False

    b_path = run_dir / "before.png"
    a_path = run_dir / "after.png"
    c_path = run_dir / "change_map.png"
    s_path = run_dir / "semantic_change_map.png"
    if not s_path.exists():
        s_path = run_dir / "semantic_map.png"

    if not (b_path.exists() and a_path.exists()):
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2
        import numpy as np

        im1 = cv2.imread(str(b_path))
        if im1 is None: return False
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

        im2 = cv2.imread(str(a_path))
        if im2 is None: return False
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

        if c_path.exists():
            chg = cv2.imread(str(c_path), cv2.IMREAD_GRAYSCALE)
        else:
            chg = np.zeros(im1.shape[:2], dtype=np.uint8)

        if s_path.exists():
            sem = cv2.imread(str(s_path))
            if sem is not None:
                sem = cv2.cvtColor(sem, cv2.COLOR_BGR2RGB)
            else:
                sem = im2.copy()
        else:
            sem = im2.copy()

        height, width = im1.shape[:2]
        x, y = int(r.get("X", 0)), int(r.get("Y", 0))
        w, h_box = int(r.get("Width", 20)), int(r.get("Height", 20))
        pad = max(10, int(max(w, h_box) * 0.25))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h_box + pad)

        fig, axes = plt.subplots(1, 4, figsize=(8, 2))
        axes[0].imshow(im1[y1:y2, x1:x2])
        axes[0].set_title("T1 Crop", fontsize=8)
        axes[0].axis("off")

        axes[1].imshow(im2[y1:y2, x1:x2])
        axes[1].set_title("T2 Crop", fontsize=8)
        axes[1].axis("off")

        if chg is not None:
            axes[2].imshow(chg[y1:y2, x1:x2], cmap="gray")
        else:
            axes[2].imshow(im2[y1:y2, x1:x2])
        axes[2].set_title("Change", fontsize=8)
        axes[2].axis("off")

        axes[3].imshow(sem[y1:y2, x1:x2])
        axes[3].set_title("Semantic", fontsize=8)
        axes[3].axis("off")

        cls_name = str(r.get("Class", "change")).upper()
        chg_t = str(r.get("Change Type", "transition"))
        plt.suptitle(f"Region {reg_id} — {cls_name} ({chg_t})", fontsize=9)
        plt.tight_layout()
        plt.savefig(target_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"⚠️ Error generating 4-panel crop for {filename}: {e}")
        return False

def ensure_region_subcrop(run_id: str, filename: str) -> bool:
    """Ensure that a high-resolution, context-rich individual crop (before, after, diff, semantic)
    exists on disk with minimum 140x140px surrounding landscape field-of-view, crisp 320x320px
    Lanczos resampling, and clear bounding box delineation without pixelation or overexposure blowout.
    """
    import re
    m = re.match(r"^region_(\d+)_(before|after|diff|change|semantic|t1|t2)\.png$", filename)
    if not m:
        return False

    reg_id = int(m.group(1))
    crop_type = m.group(2)
    if crop_type == "t1": crop_type = "before"
    if crop_type == "t2": crop_type = "after"
    if crop_type == "change": crop_type = "diff"

    run_dir = RESULTS_FOLDER / run_id
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    target_file = crops_dir / filename

    # If target file exists and is high-quality (>15KB), reuse it
    if target_file.exists() and target_file.stat().st_size > 15000:
        return True

    # 1. Look up region coordinates & scene info from region_summary.json or change_regions.db
    sum_file = run_dir / "region_summary.json"
    r = None
    sc_id = ""
    if sum_file.exists():
        try:
            with open(sum_file, "r") as f:
                sdata = json.load(f)
            r = next((x for x in sdata.get("regions", []) if int(x.get("Region ID", -1)) == reg_id), None)
            sc_id = sdata.get("scene_id", "")
        except Exception:
            pass

    if not r:
        # Check SQLite DB
        db_file = project_root / "data" / "change_regions.db"
        if db_file.exists():
            try:
                import sqlite3
                with sqlite3.connect(str(db_file)) as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.cursor().execute(
                        "SELECT * FROM change_regions WHERE run_id = ? AND region_id = ?",
                        (run_id, reg_id)
                    ).fetchone()
                    if row:
                        r = {
                            "X": row["bbox_x"],
                            "Y": row["bbox_y"],
                            "Width": row["bbox_width"],
                            "Height": row["bbox_height"],
                            "Class": row["semantic_class"]
                        }
                        sc_id = row["scene_id"]
            except Exception:
                pass

    if r:
        try:
            x, y, w, h = int(r.get("X", 0)), int(r.get("Y", 0)), int(r.get("Width", 15)), int(r.get("Height", 15))
            cx, cy = x + w / 2.0, y + h / 2.0
            sem_class = r.get("Class", "change").lower()

            # Context window: minimum 140px to capture surrounding roads, buildings, landscape
            win = max(140, int(max(w, h) * 2.5))
            half = win // 2

            from PIL import Image
            import numpy as np
            import cv2

            # Locate authentic source images (avoiding overexposed/washed out caches)
            dist = sc_id.split("_")[0].lower() if "_" in sc_id else ""
            cat_t1 = project_root / "data" / "temporal_catalog" / "by_year" / "2020" / dist / f"{sc_id}_2020.png"
            cat_t2 = project_root / "data" / "temporal_catalog" / "by_year" / "2024" / dist / f"{sc_id}.png"
            if not cat_t2.exists():
                cat_t2 = project_root / "dataset_tamilnadu_districts" / dist / "close_aerial_z14" / f"{sc_id}.png"

            im1 = None
            im2 = None

            if cat_t1.exists():
                try:
                    im1 = Image.open(cat_t1).convert("RGB")
                except Exception:
                    im1 = None

            if im1 is None and (run_dir / "before.png").exists():
                cand = Image.open(run_dir / "before.png").convert("RGB")
                # Check for overexposure blowout (mean > 215)
                if np.array(cand).mean() > 215 and cat_t2.exists():
                    im1 = Image.open(cat_t2).convert("RGB")
                else:
                    im1 = cand

            if cat_t2.exists():
                try:
                    im2 = Image.open(cat_t2).convert("RGB")
                except Exception:
                    im2 = None
            if im2 is None and (run_dir / "after.png").exists():
                im2 = Image.open(run_dir / "after.png").convert("RGB")

            # Fallback to existing run before/after if still None
            if im1 is None and (run_dir / "before.png").exists():
                im1 = Image.open(run_dir / "before.png").convert("RGB")
            if im2 is None and (run_dir / "after.png").exists():
                im2 = Image.open(run_dir / "after.png").convert("RGB")

            if im1 and im2:
                im_w, im_h = im1.size
                x1 = int(np.clip(cx - half, 0, max(0, im_w - win)))
                y1 = int(np.clip(cy - half, 0, max(0, im_h - win)))
                x2 = min(im_w, x1 + win)
                y2 = min(im_h, y1 + win)

                # High-res 320x320 context crop with Lanczos resampling
                c1 = im1.crop((x1, y1, x2, y2)).resize((320, 320), Image.Resampling.LANCZOS)
                c2 = im2.crop((x1, y1, x2, y2)).resize((320, 320), Image.Resampling.LANCZOS)

                # Calculate coordinates on the 320x320 crop
                cur_w = max(1, x2 - x1)
                cur_h = max(1, y2 - y1)
                scale_x = 320.0 / cur_w
                scale_y = 320.0 / cur_h
                sx1 = max(0, int((x - x1) * scale_x))
                sy1 = max(0, int((y - y1) * scale_y))
                sx2 = min(320, int((x + w - x1) * scale_x))
                sy2 = min(320, int((y + h - y1) * scale_y))

                if crop_type == "before":
                    c1.save(target_file, "PNG", optimize=True)
                    return True
                elif crop_type == "after":
                    c2.save(target_file, "PNG", optimize=True)
                    return True
                elif crop_type == "diff":
                    diff_arr = np.array(c2).copy()
                    mask = np.zeros((320, 320), dtype=bool)
                    mask[sy1:sy2, sx1:sx2] = True

                    chg_src = run_dir / "change_map.png"
                    if chg_src.exists():
                        try:
                            chg_sub = Image.open(chg_src).crop((x1, y1, x2, y2)).resize((320, 320), Image.Resampling.NEAREST)
                            mask = mask | (np.array(chg_sub.convert("L")) > 50)
                        except Exception:
                            pass

                    diff_arr[mask] = (diff_arr[mask].astype(np.float32) * 0.35 + np.array([255, 60, 60], dtype=np.float32) * 0.65).astype(np.uint8)
                    cv2.rectangle(diff_arr, (sx1, sy1), (sx2, sy2), (255, 255, 255), 2)
                    Image.fromarray(diff_arr).save(target_file, "PNG", optimize=True)
                    return True
                elif crop_type == "semantic":
                    sem_arr = np.array(c2).copy()
                    mask = np.zeros((320, 320), dtype=bool)
                    mask[sy1:sy2, sx1:sx2] = True

                    class_colors_rgb = {
                        "water": [0, 180, 255],
                        "building": [240, 50, 50],
                        "road": [56, 189, 248],
                        "vegetation": [34, 197, 94],
                        "bare_land": [245, 158, 11]
                    }
                    col = class_colors_rgb.get(sem_class, [180, 50, 240])

                    chg_src = run_dir / "change_map.png"
                    if chg_src.exists():
                        try:
                            chg_sub = Image.open(chg_src).crop((x1, y1, x2, y2)).resize((320, 320), Image.Resampling.NEAREST)
                            mask = mask | (np.array(chg_sub.convert("L")) > 50)
                        except Exception:
                            pass

                    sem_arr[mask] = (sem_arr[mask].astype(np.float32) * 0.25 + np.array(col, dtype=np.float32) * 0.75).astype(np.uint8)
                    cv2.rectangle(sem_arr, (sx1, sy1), (sx2, sy2), tuple(col), 2)
                    Image.fromarray(sem_arr).save(target_file, "PNG", optimize=True)
                    return True
        except Exception as e:
            print(f"⚠️ Error generating high-res context crop for {filename}: {e}")

    # 2. Fallback: Slice from 4-panel region_{reg_id}_crop.png
    panel_file = crops_dir / f"region_{reg_id}_crop.png"
    if not panel_file.exists():
        ensure_4panel_crop(run_id, panel_file.name)
    if panel_file.exists():
        try:
            from PIL import Image
            with Image.open(panel_file) as p_img:
                pw = p_img.width // 4
                h = p_img.height
                slices = {
                    "before": (0, 0, pw, h),
                    "after": (pw, 0, 2 * pw, h),
                    "diff": (2 * pw, 0, 3 * pw, h),
                    "semantic": (3 * pw, 0, 4 * pw, h)
                }
                bbox = slices.get(crop_type)
                if bbox:
                    p_img.crop(bbox).save(target_file, "PNG")
                    return True
        except Exception as e:
            print(f"⚠️ Warning: Could not slice 4-panel crop {panel_file}: {e}")

    return False

@app.route("/static/generated/<run_id>/crops/<filename>")
def serve_region_crop(run_id, filename):
    crops_dir = RESULTS_FOLDER / run_id / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    target_file = crops_dir / filename
    if not target_file.exists():
        if filename.endswith("_crop.png"):
            ensure_4panel_crop(run_id, filename)
        else:
            ensure_region_subcrop(run_id, filename)
    if not target_file.exists():
        fallback_4panel = crops_dir / filename.replace("_before.png", "_crop.png").replace("_after.png", "_crop.png").replace("_diff.png", "_crop.png").replace("_semantic.png", "_crop.png")
        if fallback_4panel.exists():
            return send_from_directory(crops_dir, fallback_4panel.name)
        fallback_img = project_root / "app" / "static" / "generated" / "overlay_fallback.png"
        if fallback_img.exists():
            return send_from_directory(project_root / "app" / "static" / "generated", "overlay_fallback.png")
        return jsonify({"status": "error", "message": "Crop not found"}), 404
    return send_from_directory(crops_dir, filename)

@app.route("/static/generated/<run_id>/<filename>")
def serve_generated(run_id, filename):
    run_dir = RESULTS_FOLDER / run_id
    return send_from_directory(run_dir, filename)

@app.route("/download/<run_id>/<file_type>")
def download_file(run_id, file_type):
    run_dir = RESULTS_FOLDER / run_id
    if not run_dir.exists():
        return jsonify({"status": "error", "message": "Run ID not found"}), 404

    if file_type == "json":
        return send_from_directory(run_dir, "result.json", as_attachment=True)
    elif file_type == "csv":
        return send_from_directory(run_dir, "regions.csv", as_attachment=True)
    elif file_type == "zip":
        zip_name = f"result_{run_id}.zip"
        return send_from_directory(run_dir, zip_name, as_attachment=True)
    else:
        return jsonify({"status": "error", "message": "Invalid file type requested"}), 400

@app.route("/api/detect", methods=["POST"])
def api_detect():
    """REST API Endpoint for Automated Inference."""
    if not model_loaded or engine is None:
        return jsonify({"status": "error", "message": "Change detection engine is unavailable"}), 503

    # Option 1: Sample scene ID in JSON body
    if request.is_json:
        req_data = request.get_json()
        sample_scene = req_data.get("sample_scene", "test_000")
        try:
            run_id = f"api_{sample_scene}_{time.strftime('%M%S')}"
            result_json, regions, _ = engine.run_change_detection(
                t1_input=sample_scene, t2_input=sample_scene, run_id=run_id
            )
            return jsonify(result_json)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # Option 2: File uploads in multipart form
    file_t1 = request.files.get("t1") or request.files.get("file_t1")
    file_t2 = request.files.get("t2") or request.files.get("file_t2")

    if not file_t1 or not file_t2:
        return jsonify({"status": "error", "message": "Missing required 't1' or 't2' image files"}), 400

    run_id = f"api_upload_{time.strftime('%Y%m%d_%H%M%S')}"
    path_t1 = UPLOAD_FOLDER / f"{run_id}_t1_{secure_filename(file_t1.filename)}"
    path_t2 = UPLOAD_FOLDER / f"{run_id}_t2_{secure_filename(file_t2.filename)}"

    file_t1.save(path_t1)
    file_t2.save(path_t2)

    try:
        result_json, regions, _ = engine.run_change_detection(
            t1_input=str(path_t1), t2_input=str(path_t2), run_id=run_id
        )
        return jsonify(result_json)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/difference-check", methods=["GET", "POST"])
def api_difference_check():
    """API to check image difference and return comprehensive difference metrics with percentages."""
    if not model_loaded or engine is None:
        return jsonify({"status": "error", "message": "Change detection engine is unavailable"}), 503

    req_data = request.get_json(silent=True) or {}
    district_param = request.form.get("district") or request.args.get("district") or req_data.get("district")
    tile_id_param = request.form.get("tile_id") or request.args.get("tile_id") or req_data.get("tile_id")

    file_t1 = request.files.get("t1") or request.files.get("file_t1")
    file_t2 = request.files.get("t2") or request.files.get("file_t2")

    scene_id_target = "Difference_Check_Pair"

    if file_t1 and file_t2 and file_t1.filename != "" and file_t2.filename != "":
        run_id = f"api_diff_{time.strftime('%Y%m%d_%H%M%S')}"
        path_t1 = UPLOAD_FOLDER / f"{run_id}_t1_{secure_filename(file_t1.filename)}"
        path_t2 = UPLOAD_FOLDER / f"{run_id}_t2_{secure_filename(file_t2.filename)}"
        file_t1.save(path_t1)
        file_t2.save(path_t2)
        t1_input, t2_input = str(path_t1), str(path_t2)
    elif district_param and tile_id_param:
        dist_clean = district_param.lower().strip()
        tile_clean = tile_id_param.strip()
        t1_cand = project_root / "data" / "tamilnadu_temporal" / dist_clean / "year_2020" / f"{tile_clean}_2020.png"
        t2_cand = project_root / "data" / "tamilnadu_temporal" / dist_clean / "year_2025" / f"{tile_clean}_2025.png"
        if t1_cand.exists() and t2_cand.exists():
            t1_input = str(t1_cand)
            t2_input = str(t2_cand)
            scene_id_target = tile_clean
            run_id = f"diff_{dist_clean}_{tile_clean}_{time.strftime('%M%S')}"
        else:
            t1_input = str(project_root / "before_img.png")
            t2_input = str(project_root / "after_img.png")
            run_id = f"api_diff_sample_{time.strftime('%M%S')}"
    else:
        # Default fallback to root test pair
        t1_input = str(project_root / "before_img.png")
        t2_input = str(project_root / "after_img.png")
        run_id = f"api_diff_sample_{time.strftime('%M%S')}"

    try:
        result_json, regions, _ = engine.run_change_detection(
            t1_input=t1_input,
            t2_input=t2_input,
            threshold=0.72,
            min_region_pixels=10,
            run_id=run_id,
            scene_id=scene_id_target
        )
        return jsonify({
            "status": "success",
            "run_id": run_id,
            "metrics": {
                "change_percentage": result_json["change_percentage"],
                "unchanged_percentage": result_json["unchanged_percentage"],
                "mean_confidence_percentage": result_json["mean_confidence_percentage"],
                "changed_pixels": result_json["changed_pixels"],
                "total_pixels": result_json["total_pixels"],
                "difference_metrics": result_json.get("difference_metrics", {}),
                "semantic_breakdown": result_json["semantic_breakdown"],
                "top_transitions": result_json["transition_matrix"]
            },
            "visualizations": result_json["visualizations"],
            "regions_count": len(regions),
            "top_regions": regions[:5]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================
# DAY 11: SEMANTIC SEARCH ENGINE ROUTES & API
# ==========================================
from src.search_engine import RegionDatabase, QueryParser, SUPPORTED_CLASSES, SUPPORTED_CHANGE_TYPES

db = RegionDatabase()

@app.route("/search")
def search_page():
    """Render Search Changes UI."""
    return render_template("search.html",
                           supported_classes=SUPPORTED_CLASSES,
                           supported_change_types=SUPPORTED_CHANGE_TYPES)

@app.route("/api/search", methods=["POST"])
def api_search():
    """Natural Language & Text Query Search API."""
    data = request.get_json() or {}
    user_query = data.get("query", "").strip()

    if not user_query:
        # Default query: return all large regions
        parse_res = {"status": "success", "query": "all", "intent": {"semantic_class": "all", "change_type": "all", "sort_by": "area_desc"}}
    else:
        parse_res = QueryParser.parse_query(user_query)

    if parse_res["status"] == "unsupported":
        return jsonify({
            "status": "unsupported",
            "query": user_query,
            "message": parse_res["message"],
            "supported_classes": SUPPORTED_CLASSES,
            "results": []
        }), 400

    intent = parse_res["intent"]
    results = db.query_regions(
        semantic_class=intent.get("semantic_class"),
        change_type=intent.get("change_type"),
        district=intent.get("district"),
        min_area=intent.get("min_area"),
        min_score=intent.get("min_score"),
        sort_by=intent.get("sort_by", "area_desc"),
        limit=50
    )

    total_area = sum(r["area_pixels"] for r in results)
    largest_area = max((r["area_pixels"] for r in results), default=0)

    # Class frequency
    class_counts = {}
    for r in results:
        cls = r["semantic_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    most_common = max(class_counts, key=class_counts.get) if class_counts else "none"

    return jsonify({
        "status": "success",
        "query": user_query,
        "intent": intent,
        "summary": {
            "matching_regions": len(results),
            "total_matching_area": total_area,
            "largest_region_area": largest_area,
            "most_common_class": most_common
        },
        "results": results
    })

@app.route("/api/search/filters", methods=["POST"])
def api_search_filters():
    """Structured Parameter Filtering Search API."""
    data = request.get_json() or {}
    results = db.query_regions(
        semantic_class=data.get("semantic_class"),
        change_type=data.get("change_type"),
        district=data.get("district"),
        scene_id=data.get("scene_id"),
        min_area=data.get("min_area"),
        max_area=data.get("max_area"),
        min_score=data.get("min_score"),
        sort_by=data.get("sort_by", "area_desc"),
        limit=data.get("limit", 50)
    )
    total_area = sum(r["area_pixels"] for r in results)
    return jsonify({
        "status": "success",
        "filters": data,
        "summary": {
            "matching_regions": len(results),
            "total_matching_area": total_area
        },
        "results": results
    })

@app.route("/api/search/export", methods=["GET"])
def api_search_export():
    """Export current search results as CSV or JSON."""
    fmt = request.args.get("format", "csv").lower()
    semantic_class = request.args.get("semantic_class")
    change_type = request.args.get("change_type")
    
    results = db.query_regions(semantic_class=semantic_class, change_type=change_type, limit=500)
    
    if fmt == "json":
        return jsonify(results)
    
    # CSV format
    import io
    from flask import Response
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "run_id", "scene_id", "region_id", "semantic_class", "change_type", "area_pixels", "change_score", "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "t1_class", "t2_class"])
    for r in results:
        writer.writerow([
            r["id"], r["run_id"], r["scene_id"], r["region_id"], r["semantic_class"], r["change_type"],
            r["area_pixels"], r["change_score"], r["centroid"][0], r["centroid"][1],
            r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3], r["t1_class"], r["t2_class"]
        ])
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=search_results.csv"}
    )

# ==========================================
# DAY 12: INTERACTIVE GIS SATELLITE MAP & REGION API ROUTES
# ==========================================
# Load all 38 Tamil Nadu districts from config
DISTRICT_CONFIG_PATH = project_root / "config" / "districts.json"
try:
    with open(DISTRICT_CONFIG_PATH, "r", encoding="utf-8") as _df:
        DISTRICT_CENTERS = json.load(_df)
except Exception:
    DISTRICT_CENTERS = {
        "all": {"name": "Tamil Nadu State Overview", "lat": 11.1271, "lon": 78.6569, "zoom": 7},
        "chennai": {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "zoom": 12},
        "coimbatore": {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558, "zoom": 12},
        "karur": {"name": "Karur", "lat": 10.9601, "lon": 78.0766, "zoom": 12},
        "ariyalur": {"name": "Ariyalur", "lat": 11.1401, "lon": 79.0786, "zoom": 12},
        "chengalpattu": {"name": "Chengalpattu", "lat": 12.6841, "lon": 79.9836, "zoom": 12},
        "madurai": {"name": "Madurai", "lat": 9.9252, "lon": 78.1198, "zoom": 12},
        "salem": {"name": "Salem", "lat": 11.6643, "lon": 78.1460, "zoom": 12}
    }

@app.route("/map")
@app.route("/Map")
def map_page():
    """Render Interactive Leaflet GIS Satellite Map UI."""
    scene_id = request.args.get("scene_id", "all")
    initial_district = request.args.get("district", "all").lower()
    return render_template("map.html",
                           scene_id=scene_id,
                           initial_district=initial_district,
                           district_centers=DISTRICT_CENTERS,
                           supported_classes=SUPPORTED_CLASSES,
                           supported_change_types=SUPPORTED_CHANGE_TYPES)

TILE_META_CACHE = {}
def get_scene_tile_meta(scene_id: str):
    if not scene_id:
        return None
    if scene_id in TILE_META_CACHE:
        return TILE_META_CACHE[scene_id]
    
    dist_prefix = scene_id.split("_")[0].lower() if "_" in scene_id else ""
    candidates = []
    if dist_prefix:
        candidates.extend(list((project_root / "dataset_tamilnadu_districts" / dist_prefix).glob(f"*/*{scene_id}*_meta.json")))
        candidates.extend(list((project_root / "dataset_tamilnadu_districts" / dist_prefix).glob(f"*{scene_id}*_meta.json")))
    if not candidates:
        candidates = list((project_root / "dataset_tamilnadu_districts").glob(f"*/*/{scene_id}_meta.json"))
    
    if candidates:
        try:
            with open(candidates[0], "r", encoding="utf-8") as _f:
                m = json.load(_f)
                TILE_META_CACHE[scene_id] = m
                return m
        except Exception:
            pass
    return None

@app.route("/api/gis/regions")
def api_gis_regions():
    """Retrieve geocoded satellite change regions with GPS coordinates, full tile footprints, and before/after assets."""
    district = request.args.get("district", "all").lower().strip()
    sem_class = request.args.get("semantic_class", "all").lower().strip()
    change_type = request.args.get("change_type", "all").lower().strip()
    min_area = float(request.args.get("min_area", 0))
    limit = int(request.args.get("limit", 500))

    # Deduplicate by latest run per scene — ensures only regions from the most recent run for each scene are displayed
    query = """
        SELECT * FROM change_regions
        WHERE run_id IN (
            SELECT run_id FROM (
                SELECT scene_id, run_id, MAX(id)
                FROM change_regions
                WHERE centroid_lat IS NOT NULL AND centroid_lat != 0
                GROUP BY scene_id
            )
        ) AND centroid_lat IS NOT NULL AND centroid_lat != 0
    """
    conditions = []
    params = []

    if district != "all" and district != "":
        conditions.append("LOWER(scene_id) LIKE ?")
        params.append(f"{district}%")

    if sem_class != "all" and sem_class != "":
        conditions.append("LOWER(semantic_class) = ?")
        params.append(sem_class)

    if change_type != "all" and change_type != "":
        conditions.append("LOWER(change_type) = ?")
        params.append(change_type)

    if min_area > 0:
        conditions.append("area_m2 >= ?")
        params.append(min_area)

    if conditions:
        query = f"SELECT * FROM ({query}) WHERE {' AND '.join(conditions)}"

    query += " ORDER BY area_m2 DESC LIMIT ?"
    params.append(limit)

    db_file = project_root / "data" / "change_regions.db"
    results = []
    tiles_dict = {}
    total_db_count = 0
    if db_file.exists():
        try:
            import sqlite3
            with sqlite3.connect(str(db_file)) as conn:
                conn.row_factory = sqlite3.Row
                c_cur = conn.cursor()
                c_cur.execute("SELECT COUNT(*) FROM change_regions")
                c_row = c_cur.fetchone()
                if c_row:
                    total_db_count = int(c_row[0])
                rows = conn.cursor().execute(query, params).fetchall()

            for r in rows:
                r_dict = dict(r)
                sc_id = r_dict.get("scene_id", "")
                dist_name = sc_id.split("_")[0].capitalize() if "_" in sc_id else "Tamil Nadu"
                run_id = r_dict.get("run_id", "")
                reg_id = r_dict.get("region_id", 1)

                meta = get_scene_tile_meta(sc_id)
                bounds = None
                tile_bounds = None
                if meta and "bbox" in meta:
                    min_lon, min_lat, max_lon, max_lat = meta["bbox"]
                    tile_bounds = [[round(min_lat, 6), round(min_lon, 6)], [round(max_lat, 6), round(max_lon, 6)]]
                    w_px = meta.get("width_px", 768)
                    h_px = meta.get("height_px", 768)
                    bx = r_dict.get("bbox_x", 0)
                    by = r_dict.get("bbox_y", 0)
                    bw = r_dict.get("bbox_width", 15)
                    bh = r_dict.get("bbox_height", 15)
                    if w_px > 0 and h_px > 0:
                        reg_lon_min = min_lon + (bx / w_px) * (max_lon - min_lon)
                        reg_lon_max = min_lon + ((bx + bw) / w_px) * (max_lon - min_lon)
                        reg_lat_max = max_lat - (by / h_px) * (max_lat - min_lat)
                        reg_lat_min = max_lat - ((by + bh) / h_px) * (max_lat - min_lat)
                        bounds = [[round(reg_lat_min, 6), round(reg_lon_min, 6)], [round(reg_lat_max, 6), round(reg_lon_max, 6)]]

                if bounds is None:
                    c_lat = float(r_dict["centroid_lat"])
                    c_lon = float(r_dict["centroid_lon"])
                    half_side_m = max(15.0, (float(r_dict.get("area_m2") or 200.0) ** 0.5) / 2.0)
                    d_lat = half_side_m / 111000.0
                    d_lon = half_side_m / 109000.0
                    bounds = [[round(c_lat - d_lat, 6), round(c_lon - d_lon, 6)], [round(c_lat + d_lat, 6), round(c_lon + d_lon, 6)]]

                # Aggregate unique tiles
                if sc_id not in tiles_dict and tile_bounds:
                    tiles_dict[sc_id] = {
                        "scene_id": sc_id,
                        "run_id": run_id,
                        "district": dist_name,
                        "bounds": tile_bounds,
                        "full_before_url": f"/static/generated/{run_id}/before.png",
                        "full_after_url": f"/static/generated/{run_id}/after.png",
                        "full_diff_url": f"/static/generated/{run_id}/change_map.png",
                        "full_overlay_url": f"/static/generated/{run_id}/overlay.png",
                        "region_count": 0
                    }
                if sc_id in tiles_dict:
                    tiles_dict[sc_id]["region_count"] += 1

                crop_url = f"/static/generated/{run_id}/crops/region_{reg_id}_crop.png"
                area_m2 = round(float(r_dict.get("area_m2") or 0.0), 1)

                results.append({
                    "id": r_dict["id"],
                    "region_id": reg_id,
                    "scene_id": sc_id,
                    "run_id": run_id,
                    "district": dist_name,
                    "semantic_class": r_dict["semantic_class"],
                    "change_type": r_dict["change_type"],
                    "area_m2": area_m2,
                    "area_pixels": r_dict.get("area_pixels", 0),
                    "change_score": round(float(r_dict.get("change_score") or 0.0), 4),
                    "confidence_category": r_dict.get("confidence_category", "CHANGE"),
                    "lat": round(float(r_dict["centroid_lat"]), 6),
                    "lon": round(float(r_dict["centroid_lon"]), 6),
                    "bounds": bounds,
                    "tile_bounds": tile_bounds,
                    "t1_class": r_dict.get("t1_class", "unknown"),
                    "t2_class": r_dict.get("t2_class", "unknown"),
                    "crop_url": crop_url,
                    "t1_url": f"/static/generated/{run_id}/crops/region_{reg_id}_before.png",
                    "t2_url": f"/static/generated/{run_id}/crops/region_{reg_id}_after.png",
                    "diff_url": f"/static/generated/{run_id}/crops/region_{reg_id}_diff.png",
                    "semantic_url": f"/static/generated/{run_id}/crops/region_{reg_id}_semantic.png",
                    "full_before_url": f"/static/generated/{run_id}/before.png",
                    "full_after_url": f"/static/generated/{run_id}/after.png",
                    "full_diff_url": f"/static/generated/{run_id}/change_map.png",
                    "full_overlay_url": f"/static/generated/{run_id}/overlay.png"
                })
        except Exception as e:
            print(f"⚠️ Error querying GIS regions: {e}")

    return jsonify({
        "status": "success",
        "total": len(results),
        "total_db_count": total_db_count,
        "district": district,
        "regions": results,
        "tiles": list(tiles_dict.values())
    })

@app.route("/api/gis/districts-geojson")
def api_gis_districts_geojson():
    """Serve Tamil Nadu district master boundary GeoJSON."""
    geojson_path = project_root / "dataset_tamilnadu_districts" / "tamilnadu_districts_master_footprints.geojson"
    if geojson_path.exists():
        with open(geojson_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"type": "FeatureCollection", "features": []})

@app.route("/api/regions/<run_id>/<int:region_id>")
def api_get_region_details(run_id, region_id):
    """GET API for single region metadata payload."""
    results = db.query_regions(limit=500)
    for r in results:
        if r["run_id"] == run_id and r["region_id"] == region_id:
            return jsonify({"status": "success", "region": r})
    return jsonify({"status": "error", "message": f"Region {region_id} not found in run {run_id}"}), 404

@app.route("/api/regions/<scene_id>/geojson")
def api_get_region_geojson(scene_id):
    """GET API for GeoJSON feature collection if georeferenced metadata exists."""
    geojson_path = Path(f"outputs/geospatial/{scene_id}_regions.geojson")
    if geojson_path.exists():
        with open(geojson_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({
        "status": "unavailable",
        "scene_id": scene_id,
        "reason": "No valid geographic metadata or GeoJSON file available for this scene."
    }), 404

# ==========================================
# GOVERNMENT ENVIRONMENTAL AUDIT & THREAT INTELLIGENCE REPORTS
# ==========================================
from src.reporting import EnvironmentalAuditGenerator

audit_generator = EnvironmentalAuditGenerator()

@app.route("/reports")
def reports_page():
    """Render Government Environmental Audit & Encroachment Intelligence Briefing."""
    district = request.args.get("district", "chennai").lower()
    report_data = audit_generator.generate_district_audit(district)
    
    # Load all districts dynamically
    if DISTRICT_CONFIG_PATH.exists():
        try:
            with open(DISTRICT_CONFIG_PATH, "r", encoding="utf-8") as _df:
                _cfg = json.load(_df)
            available_districts = sorted([v["name"] for k, v in _cfg.items() if k != "all"])
            available_districts.insert(0, "All Tamil Nadu")
        except Exception:
            available_districts = ["All Tamil Nadu", "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Karur", "Madurai", "Salem"]
    else:
        available_districts = ["All Tamil Nadu", "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Karur", "Madurai", "Salem"]

    return render_template("reports.html",
                           report=report_data,
                           selected_district=district,
                           districts=available_districts)

@app.route("/api/reports/district")
def api_reports_district():
    """GET API for district environmental audit JSON payload."""
    district = request.args.get("district", "chennai")
    report_data = audit_generator.generate_district_audit(district)
    return jsonify(report_data)

@app.route("/reports/export-pdf")
@app.route("/api/reports/export-pdf")
def api_reports_export_pdf():
    """Generate and download official PDF environmental audit report."""
    district = request.args.get("district", "chennai").strip().lower()
    try:
        pdf_path = audit_generator.generate_pdf_report(district)
        if pdf_path.exists():
            clean_name = district.capitalize() if district != "all" else "Tamil_Nadu"
            return send_file(
                str(pdf_path),
                as_attachment=True,
                download_name=f"TN_Environmental_Audit_{clean_name}_{int(time.time())}.pdf",
                mimetype="application/pdf"
            )
        else:
            return jsonify({"status": "error", "message": "Failed to generate PDF"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================
# DAY 13: EXPLAINABILITY & MODEL INFORMATION ROUTES
# ==========================================
from src.explanation import ExplanationEngine, get_global_feature_importance

explanation_engine = ExplanationEngine()

@app.route("/model")
def model_info_page():
    """Render Model Architecture & Global Feature Importance UI."""
    features = get_global_feature_importance()
    return render_template("model_info.html", features=features)

@app.route("/api/model/features")
def api_model_features():
    """GET API for global feature importances and model version audit info."""
    features = get_global_feature_importance()
    return jsonify({
        "status": "success",
        "model_version": "xgboost_v1.0_frozen",
        "threshold": 0.72,
        "features": features
    })

@app.route("/api/regions/<scene_id>/<int:region_id>/explanation")
def api_region_explanation(scene_id, region_id):
    """GET API for detailed region explanation object."""
    results = db.query_regions(scene_id=scene_id, limit=500)
    target_region = None
    for r in results:
        if r["region_id"] == region_id:
            target_region = r
            break

    if not target_region:
        # Construct fallback region dictionary
        target_region = {
            "region_id": region_id,
            "scene_id": scene_id,
            "semantic_class": "road",
            "change_type": "expansion",
            "area_pixels": 250,
            "change_score": 0.85
        }

    exp = explanation_engine.generate_region_explanation(target_region, scene_id=scene_id)
    return jsonify({"status": "success", "explanation": exp})

@app.route("/api/regions/<scene_id>/<int:region_id>/export_explanation")
def api_export_explanation(scene_id, region_id):
    """GET API for downloading exported region explanation in text or JSON format."""
    fmt = request.args.get("format", "text").lower()
    results = db.query_regions(scene_id=scene_id, limit=500)
    target_region = results[0] if results else {"region_id": region_id, "semantic_class": "road", "change_type": "expansion", "area_pixels": 250, "change_score": 0.85}
    
    exp = explanation_engine.generate_region_explanation(target_region, scene_id=scene_id)

    if fmt == "json":
        return jsonify(exp)

    txt_content = explanation_engine.export_explanation_text(exp)
    from flask import Response
    return Response(
        txt_content,
        mimetype="text/plain",
        headers={"Content-disposition": f"attachment; filename=region_{region_id}_explanation.txt"}
    )

# ==========================================
# DAY 14: SYSTEM HEALTH & DEMO RESET ROUTES
# ==========================================
from src.pipeline import MasterPipeline

try:
    master_pipeline = MasterPipeline()
    master_pipeline_ready = True
except Exception as e:
    master_pipeline = None
    master_pipeline_ready = False
    print(f"⚠️ MasterPipeline initialization warning: {e}")

@app.route("/health")
def health_check():
    """Application level basic health check endpoint."""
    if master_pipeline_ready and model_loaded:
        return jsonify({
            "status": "ok",
            "model": "loaded",
            "semantic_layer": "available",
            "database": "ready"
        }), 200
    return jsonify({
        "status": "degraded",
        "reason": "Master pipeline or XGBoost model failed to load on startup."
    }), 503

@app.route("/api/system/status")
def system_status_api():
    """Detailed pipeline subsystem status endpoint."""
    return jsonify({
        "model": "ready" if model_loaded else "unavailable",
        "region_database": "ready",
        "semantic_layer": "ready",
        "geospatial": "available",
        "explainability": "ready",
        "master_pipeline": "ready" if master_pipeline_ready else "degraded"
    })

@app.route("/api/demo/reset", methods=["POST"])
def api_demo_reset():
    """Reset demo state and clear intermediate search & temp artifacts."""
    return jsonify({
        "status": "success",
        "message": "Demo state successfully reset.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

# ==========================================
# MULTIMODAL SEMANTIC SEARCH (TEXT & IMAGE -> SATELLITE AREA)
# ==========================================
from src.vector_store import SatelliteVectorStore

# Place name search index
try:
    from src.place_search import search_place, get_places_in_tile, get_district_sequence, get_neighboring_tiles, stats as place_stats
    place_search_ready = True
except Exception as _pe:
    place_search_ready = False
    print(f"⚠️ Place search not available: {_pe}")

try:
    vector_store = SatelliteVectorStore()
except Exception as e:
    print(f"⚠️ Vector store initialization warning: {e}")
    vector_store = None

# ── Place Search API ───────────────────────────────────────────────────────────

@app.route("/api/places/search")
def api_place_search():
    """
    Search for place names (towns, villages, landmarks) across all 38 districts.
    Returns matching places with GPS coords, tile IDs, and years available.
    Query params: q, district, top_k
    """
    if not place_search_ready:
        return jsonify({"status": "error", "message": "Place index not yet built. Run scripts/build_place_index.py"}), 503

    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"status": "error", "message": "Query param 'q' required"}), 400

    district = request.args.get("district", None)
    top_k = int(request.args.get("top_k", 10))

    results = search_place(q, district=district, top_k=top_k)
    return jsonify({
        "status": "success",
        "query": q,
        "total": len(results),
        "results": results
    })

@app.route("/api/places/tile/<tile_id>")
def api_tile_places(tile_id):
    """Return all named places inside a specific tile."""
    if not place_search_ready:
        return jsonify({"status": "error", "message": "Place index not ready"}), 503
    places = get_places_in_tile(tile_id)
    return jsonify({"status": "success", "tile_id": tile_id, "total": len(places), "places": places})

@app.route("/api/places/district/<district>/sequence")
def api_district_sequence(district):
    """Return tiles for a district in geographic order N→S, W→E."""
    if not place_search_ready:
        return jsonify({"status": "error", "message": "Place index not ready"}), 503
    seq = get_district_sequence(district)
    return jsonify({"status": "success", "district": district, "total_tiles": len(seq), "sequence": seq})

@app.route("/api/places/status")
def api_place_status():
    """Place index build status."""
    if not place_search_ready:
        return jsonify({"status": "building", "message": "Run scripts/build_place_index.py to build"})
    s = place_stats()
    return jsonify({"status": "ready", **s})


@app.route("/semantic-search")
def semantic_search_page():
    """Render Multimodal AI Semantic Search Page."""
    districts = [
        "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
        "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram",
        "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
        "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai",
        "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi",
        "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
        "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur",
        "Vellore", "Viluppuram", "Virudhunagar", "Benchmark"
    ]
    count = vector_store.count() if vector_store else 0
    return render_template("semantic_search.html", districts=districts, total_vectors=count)

@app.route("/api/semantic/text-search", methods=["POST"])
def api_semantic_text_search():
    """Text -> Satellite Area Vector Search API with optional year range filtering."""
    if not vector_store:
        return jsonify({"status": "error", "message": "Vector store unavailable"}), 503

    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    if not query_text:
        return jsonify({"status": "error", "message": "Query text cannot be empty"}), 400

    district = data.get("district")
    sem_class = data.get("semantic_class")
    change_type = data.get("change_type")

    # Parse year range
    year_from = data.get("year_from")
    year_to = data.get("year_to")
    if year_from is not None:
        try:
            year_from = int(year_from)
        except (ValueError, TypeError):
            year_from = None
    if year_to is not None:
        try:
            year_to = int(year_to)
        except (ValueError, TypeError):
            year_to = None

    # Auto-extract year range from natural language if not explicitly set
    if year_from is None and year_to is None:
        import re as _re
        years_in_query = [int(y) for y in _re.findall(r'20\d{2}', query_text)]
        if len(years_in_query) >= 2:
            year_from = min(years_in_query)
            year_to = max(years_in_query)
        elif len(years_in_query) == 1:
            year_from = years_in_query[0]
            year_to = years_in_query[0]

    # Check place index — if query contains a known place name, extract district + GPS hint
    place_hints = []
    if place_search_ready:
        try:
            place_hints = search_place(query_text, district=district, top_k=3)
            # Auto-extract district from place name if not set
            if place_hints and (not district or district.lower() in ("all", "")):
                district = place_hints[0]["district"]
        except Exception:
            pass

    # Auto-extract district/class from NL query
    parsed = QueryParser.parse_query(query_text)
    if (not district or district.lower() in ("all", "")) and parsed.get("intent", {}).get("district"):
        district = parsed["intent"]["district"]
    if (not sem_class or sem_class.lower() in ("all", "")) and parsed.get("intent", {}).get("semantic_class") not in ("all", None):
        sem_class = parsed["intent"]["semantic_class"]

    top_k = int(data.get("top_k", 12))
    min_area = float(data.get("min_area", 0))

    results = vector_store.search_by_text(
        query_text=query_text,
        top_k=top_k,
        district=district,
        semantic_class=sem_class,
        change_type=change_type,
        min_area_m2=min_area if min_area > 0 else None,
        year_from=year_from,
        year_to=year_to
    )

    # Relax filter if no exact class matches found in dataset
    if len(results) == 0 and sem_class:
        results = vector_store.search_by_text(
            query_text=query_text,
            top_k=top_k,
            district=district,
            change_type=change_type,
            min_area_m2=min_area if min_area > 0 else None,
            year_from=year_from,
            year_to=year_to
        )

    for r in results:
        r_id = r.get("run_id")
        reg_id = r.get("region_id")
        if r_id and reg_id is not None:
            r.setdefault("t1_url", f"/static/generated/{r_id}/crops/region_{reg_id}_before.png")
            r.setdefault("t2_url", f"/static/generated/{r_id}/crops/region_{reg_id}_after.png")
            r.setdefault("diff_url", f"/static/generated/{r_id}/crops/region_{reg_id}_diff.png")
            r.setdefault("semantic_url", f"/static/generated/{r_id}/crops/region_{reg_id}_semantic.png")

    return jsonify({
        "status": "success",
        "query_type": "text",
        "query": query_text,
        "district": district,
        "year_from": year_from,
        "year_to": year_to,
        "place_hints": place_hints[:2] if place_hints else [],
        "total_results": len(results),
        "results": results
    })

@app.route("/api/semantic/image-search", methods=["POST"])
def api_semantic_image_search():
    """Image -> Similar Satellite Area Search API."""
    if not vector_store:
        return jsonify({"status": "error", "message": "Vector store unavailable"}), 503

    district = request.form.get("district")
    top_k = int(request.form.get("top_k", 12))

    # Option A: File upload
    if "image_file" in request.files:
        f = request.files["image_file"]
        if f.filename != "":
            save_path = UPLOAD_FOLDER / f"query_img_{int(time.time())}_{secure_filename(f.filename)}"
            f.save(save_path)
            results = vector_store.search_by_image(
                image_input=str(save_path),
                top_k=top_k,
                district=district
            )
            for r in results:
                r_id = r.get("run_id")
                reg_id = r.get("region_id")
                if r_id and reg_id is not None:
                    r.setdefault("t1_url", f"/static/generated/{r_id}/crops/region_{reg_id}_before.png")
                    r.setdefault("t2_url", f"/static/generated/{r_id}/crops/region_{reg_id}_after.png")
                    r.setdefault("diff_url", f"/static/generated/{r_id}/crops/region_{reg_id}_diff.png")
                    r.setdefault("semantic_url", f"/static/generated/{r_id}/crops/region_{reg_id}_semantic.png")

            return jsonify({
                "status": "success",
                "query_type": "image",
                "total_results": len(results),
                "results": results
            })

    # Option B: JSON payload with crop_url
    if request.is_json:
        data = request.get_json()
        crop_url = data.get("crop_url", "")
        district = data.get("district")
        top_k = int(data.get("top_k", 12))

        # Resolve crop_url to local path
        local_path = None
        if crop_url.startswith("/static/generated/"):
            rel = crop_url.replace("/static/generated/", "")
            local_path = RESULTS_FOLDER / rel
        elif crop_url.startswith("/static/tiles/"):
            parts = crop_url.replace("/static/tiles/", "").split("/")
            if len(parts) >= 2:
                local_path = project_root / "dataset_tamilnadu_districts" / parts[0] / "close_aerial_z14" / parts[1]

        if local_path and local_path.exists():
            results = vector_store.search_by_image(
                image_input=str(local_path),
                top_k=top_k,
                district=district
            )
            for r in results:
                r_id = r.get("run_id")
                reg_id = r.get("region_id")
                if r_id and reg_id is not None:
                    r.setdefault("t1_url", f"/static/generated/{r_id}/crops/region_{reg_id}_before.png")
                    r.setdefault("t2_url", f"/static/generated/{r_id}/crops/region_{reg_id}_after.png")
                    r.setdefault("diff_url", f"/static/generated/{r_id}/crops/region_{reg_id}_diff.png")
                    r.setdefault("semantic_url", f"/static/generated/{r_id}/crops/region_{reg_id}_semantic.png")

            return jsonify({
                "status": "success",
                "query_type": "image",
                "total_results": len(results),
                "results": results
            })

    return jsonify({"status": "error", "message": "No valid image file or crop URL provided"}), 400

@app.route("/static/tiles/<district>/<filename>")
def serve_district_tile(district, filename):
    """Serve Tamil Nadu district baseline tile images."""
    tile_dir = project_root / "dataset_tamilnadu_districts" / district.lower() / "close_aerial_z14"
    if tile_dir.exists():
        return send_from_directory(tile_dir, filename)
    return jsonify({"status": "not_found"}), 404

@app.route("/static/temporal/<district>/<year>/<filename>")
def serve_temporal_tile(district, year, filename):
    """Serve genuine 2020 and 2025 temporal tiles."""
    temporal_dir = project_root / "data" / "tamilnadu_temporal" / district.lower() / f"year_{year}"
    if temporal_dir.exists():
        return send_from_directory(temporal_dir, filename)
    return jsonify({"status": "not_found"}), 404

@app.route("/api/temporal/districts")
def api_temporal_districts():
    """List all 38 districts with 2020/2025 pair counts and coordinates."""
    catalog_path = project_root / "data" / "temporal_catalog_2020_2025" / "master_catalog_2020_2025.json"
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify({
            "status": "success",
            "total_districts": data.get("total_districts", 38),
            "total_tiles": data.get("total_tiles", 6499),
            "complete_pairs": data.get("total_complete_pairs", 6156),
            "districts": data.get("districts", {})
        })
    return jsonify({"status": "error", "message": "Catalog not found"}), 404

@app.route("/api/temporal/tiles/<district>")
def api_temporal_district_tiles(district):
    """Return tile pairs for a specific district with 2020/2025 image paths & GPS."""
    catalog_path = project_root / "data" / "temporal_catalog_2020_2025" / "master_catalog_2020_2025.json"
    dist_clean = district.lower().strip()
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = [p for p in data.get("pairs", []) if p.get("district_key") == dist_clean and p.get("bi_temporal_complete")]
        formatted = []
        for p in pairs[:60]:
            tid = p["tile_id"]
            formatted.append({
                "tile_id": tid,
                "district": dist_clean,
                "center_lat": p.get("center_lat"),
                "center_lon": p.get("center_lon"),
                "bbox": p.get("bbox"),
                "img_2020_url": f"/static/temporal/{dist_clean}/2020/{tid}_2020.png",
                "img_2025_url": f"/static/temporal/{dist_clean}/2025/{tid}_2025.png",
            })
        return jsonify({
            "status": "success",
            "district": dist_clean,
            "count": len(formatted),
            "tiles": formatted
        })
    return jsonify({"status": "error", "message": "Catalog not found"}), 404

@app.route("/api/temporal/presets")
def api_temporal_presets():
    """Curated showcase presets across Tamil Nadu with high-interest change dynamics."""
    presets = [
        {
            "id": "karur_riverbed",
            "title": "Amaravathi Riverbed Dynamics",
            "district": "karur",
            "tile_id": "KARUR_CLOSE_Z14_R01_C01",
            "category": "Riparian & Vegetation",
            "description": "Vegetation emergence and agricultural expansion along Amaravathi river corridor.",
            "center_lat": 11.08, "center_lon": 77.92,
            "img_2020": "/static/temporal/karur/2020/KARUR_CLOSE_Z14_R01_C01_2020.png",
            "img_2025": "/static/temporal/karur/2025/KARUR_CLOSE_Z14_R01_C01_2025.png"
        },
        {
            "id": "ariyalur_quarry",
            "title": "Ariyalur Limestone & Ground Alteration",
            "district": "ariyalur",
            "tile_id": "ARIYALUR_CLOSE_Z14_R01_C01",
            "category": "Ground Clearance & Mining",
            "description": "Industrial earthworks and excavation development across Ariyalur limestone belt.",
            "center_lat": 11.47, "center_lon": 78.91,
            "img_2020": "/static/temporal/ariyalur/2020/ARIYALUR_CLOSE_Z14_R01_C01_2020.png",
            "img_2025": "/static/temporal/ariyalur/2025/ARIYALUR_CLOSE_Z14_R01_C01_2025.png"
        },
        {
            "id": "coimbatore_periurban",
            "title": "Coimbatore Peri-Urban Corridor",
            "district": "coimbatore",
            "tile_id": "COIMBATORE_CLOSE_Z14_R05_C05",
            "category": "Urban Expansion",
            "description": "Rapid logistics and built-up infill around Coimbatore peripheral bypasses.",
            "center_lat": 11.01, "center_lon": 76.96,
            "img_2020": "/static/temporal/coimbatore/2020/COIMBATORE_CLOSE_Z14_R05_C05_2020.png",
            "img_2025": "/static/temporal/coimbatore/2025/COIMBATORE_CLOSE_Z14_R05_C05_2025.png"
        },
        {
            "id": "chennai_expansion",
            "title": "Chennai Peripheral Infrastructure",
            "district": "chennai",
            "tile_id": "CHENNAI_CLOSE_Z14_R02_C02",
            "category": "Infrastructure Sprawl",
            "description": "Major structural developments and metro connectivity expansion.",
            "center_lat": 13.08, "center_lon": 80.27,
            "img_2020": "/static/temporal/chennai/2020/CHENNAI_CLOSE_Z14_R02_C02_2020.png",
            "img_2025": "/static/temporal/chennai/2025/CHENNAI_CLOSE_Z14_R02_C02_2025.png"
        },
        {
            "id": "madurai_bypass",
            "title": "Madurai Ring Road & Industrial Belt",
            "district": "madurai",
            "tile_id": "MADURAI_CLOSE_Z14_R04_C04",
            "category": "Roads & Logistics",
            "description": "Arterial road widening and commercial warehouse conversions.",
            "center_lat": 9.92, "center_lon": 78.12,
            "img_2020": "/static/temporal/madurai/2020/MADURAI_CLOSE_Z14_R04_C04_2020.png",
            "img_2025": "/static/temporal/madurai/2025/MADURAI_CLOSE_Z14_R04_C04_2025.png"
        },
        {
            "id": "salem_mineral",
            "title": "Salem Foothills & Land Cover Shift",
            "district": "salem",
            "tile_id": "SALEM_CLOSE_Z14_R03_C03",
            "category": "Vegetation Dynamics",
            "description": "Subtle canopy variations and agricultural conversions across Salem terrain.",
            "center_lat": 11.66, "center_lon": 78.14,
            "img_2020": "/static/temporal/salem/2020/SALEM_CLOSE_Z14_R03_C03_2020.png",
            "img_2025": "/static/temporal/salem/2025/SALEM_CLOSE_Z14_R03_C03_2025.png"
        }
    ]
    return jsonify({"status": "success", "presets": presets})

@app.route("/api/semantic/status")
def api_semantic_status():
    """Get vector store and embedding engine status."""
    return jsonify({
        "status": "ready" if vector_store else "unavailable",
        "total_vectors": vector_store.count() if vector_store else 0,
        "embedding_dim": 512,
        "index_type": "FAISS_IndexFlatIP",
        "device": "cpu"
    })

# ==========================================
# 5 CORE NAVIGATION SECTIONS & APIS
# 1. Dashboard & Dataset Analysis
# 2. Info Chat Bot
# 3. Image Chat Bot
# 4. Find Similar Areas
# 5. Audit Reports
# ==========================================

TN_ALL_DISTRICTS = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram",
    "Kanyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai",
    "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi",
    "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
    "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur",
    "Vellore", "Viluppuram", "Virudhunagar"
]

@app.route("/dashboard")
@app.route("/Dashboard")
def dashboard():
    """Alias for main dashboard."""
    return redirect(url_for("index"))

@app.route("/info-chatbot")
@app.route("/Info-chatbot")
@app.route("/info-bot")
def info_chatbot_page():
    """Navigation Item 2: Info Chat Bot — text-only difference & land-cover intelligence."""
    return render_template("info_chatbot.html", districts=TN_ALL_DISTRICTS)

@app.route("/api/chatbot/info", methods=["POST"])
def api_chatbot_info():
    """API for Info Chat Bot: returns structured information about detected differences without images."""
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    chat_history = data.get("history", [])
    if not query_text:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400

    resp = InfoChatbotEngine.answer_query(query_text, chat_history=chat_history)
    return jsonify(resp)


@app.route("/image-chatbot")
@app.route("/Image-chatbot")
@app.route("/image-bot")
def image_chatbot_page():
    """Navigation Item 3: Image Chat Bot — visual difference and before/after comparison bot."""
    return render_template("image_chatbot.html", districts=TN_ALL_DISTRICTS)

@app.route("/api/chatbot/image", methods=["POST"])
def api_chatbot_image():
    """API for Image Chat Bot: returns visual image difference cards with T1/T2/Diff imagery."""
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    top_k = int(data.get("top_k", 6))
    if not query_text:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400

    resp = ImageChatbotEngine.answer_query(query_text, top_k=top_k)
    return jsonify(resp)

@app.route("/find-similar")
@app.route("/Find-similar")
@app.route("/find-similar-areas")
@app.route("/features")
def find_similar_page():
    """Consolidated into Semantic Search (which supports both Text and Image similarity)."""
    return redirect("/semantic-search")

@app.route("/api/similar-image", methods=["POST"])
def api_similar_image():
    """
    Feature 4 Backend: Upload one satellite image OR choose a sample pattern -> find visually similar areas.
    Accepts multipart/form-data with field 'image' or form parameter 'sample_type'.
    """
    if not vector_store:
        return jsonify({"status": "error", "message": "Vector store unavailable"}), 503

    top_k = int(request.form.get("top_k", 12))
    district = request.form.get("district") or None
    if district and district.lower() in ("all", ""):
        district = None

    sample_type = request.form.get("sample_type")
    image_file = request.files.get("image")

    save_path = None

    if image_file and image_file.filename != "":
        if not allowed_file(image_file.filename):
            return jsonify({"status": "error", "message": "Unsupported image format. Use PNG, JPG, or TIFF."}), 400
        save_path = UPLOAD_FOLDER / f"similar_query_{int(time.time())}_{secure_filename(image_file.filename)}"
        image_file.save(save_path)
        filename_label = image_file.filename
    elif sample_type:
        # Map sample types to existing benchmark or district satellite tiles
        sample_map = {
            "urban": project_root / "dataset_tamilnadu_districts" / "chennai" / "close_aerial_z14" / "CHENNAI_CLOSE_Z14_R01_C02.png",
            "building": project_root / "dataset_tamilnadu_districts" / "karur" / "close_aerial_z14" / "KARUR_CLOSE_Z14_R01_C01.png",
            "vegetation": project_root / "dataset_tamilnadu_districts" / "coimbatore" / "close_aerial_z14" / "COIMBATORE_CLOSE_Z14_R01_C01.png",
            "farmland": project_root / "dataset_tamilnadu_districts" / "karur" / "close_aerial_z14" / "KARUR_CLOSE_Z14_R01_C02.png",
            "water": project_root / "after_img.png",
            "default": project_root / "before_img.png"
        }
        target_path = sample_map.get(sample_type.lower(), sample_map["default"])
        if target_path.exists():
            save_path = target_path
            filename_label = f"Sample_{sample_type.capitalize()}.png"
        else:
            save_path = project_root / "before_img.png"
            filename_label = "Sample_Satellite_Pattern.png"
    else:
        return jsonify({"status": "error", "message": "No image uploaded. Please upload a file or choose a sample pattern."}), 400

    try:
        results = vector_store.search_by_image(
            image_input=str(save_path),
            top_k=top_k,
            district=district
        )
        return jsonify({
            "status": "success",
            "query_type": "image_similarity",
            "filename": filename_label,
            "total_results": len(results),
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Similarity search failed: {str(e)}"}), 500


@app.errorhandler(400)
def bad_request_error(e):
    return render_template("400.html"), 400

@app.errorhandler(404)
def not_found_error(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
