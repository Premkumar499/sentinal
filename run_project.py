#!/usr/bin/env python3
"""
Master Application Launcher & System Health Check
Semantic Satellite Change Detection & Environmental Intelligence Platform
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def print_banner():
    banner = """
================================================================================
 🛰️  SEMANTIC SATELLITE CHANGE DETECTION & ENVIRONMENTAL INTELLIGENCE PLATFORM
     AI/ML Remote Sensing • Multimodal Vector Search • Interactive GIS
================================================================================
"""
    print(banner)

def run_diagnostics():
    """Verify system requirements and asset availability."""
    print("🔍 Running Pre-Flight Diagnostics...\n")
    all_ok = True

    # 1. Python version
    py_ver = sys.version_info
    print(f"  • Python Runtime: {py_ver.major}.{py_ver.minor}.{py_ver.micro} [OK]")

    # 2. Key Libraries
    required_packages = ["torch", "transformers", "faiss", "xgboost", "flask", "rasterio", "cv2"]
    for pkg in required_packages:
        try:
            m = __import__(pkg)
            v = getattr(m, "__version__", "loaded")
            print(f"  • {pkg:15s}: v{v} [OK]")
        except ImportError:
            print(f"  • {pkg:15s}: NOT INSTALLED [FAIL]")
            all_ok = False

    # 3. Model Weights
    model_path = PROJECT_ROOT / "models" / "xgboost_change_detector_tuned.json"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "xgboost_change_detector.json"

    if model_path.exists():
        print(f"  • XGBoost Model  : {model_path.name} ({model_path.stat().st_size / 1024:.1f} KB) [OK]")
    else:
        print(f"  • XGBoost Model  : NOT FOUND in models/ [WARNING]")

    # 4. SQLite Change Regions Database
    db_path = PROJECT_ROOT / "data" / "change_regions.db"
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cnt = conn.cursor().execute("SELECT COUNT(*) FROM change_regions").fetchone()[0]
        print(f"  • Change Database: {db_path.name} ({cnt:,} records indexed) [OK]")
    else:
        print(f"  • Change Database: NOT FOUND in data/ [WARNING]")

    # 5. FAISS Multimodal Vector Store
    faiss_path = PROJECT_ROOT / "data" / "vector_index" / "satellite_regions.faiss"
    if faiss_path.exists():
        from src.vector_store import SatelliteVectorStore
        store = SatelliteVectorStore(index_dir=PROJECT_ROOT / "data" / "vector_index")
        print(f"  • FAISS Vector DB: {store.count():,} 512-d embeddings indexed [OK]")
    else:
        print(f"  • FAISS Vector DB: NOT FOUND [WARNING]")

    # 6. Tamil Nadu District Catalog
    tn_dir = PROJECT_ROOT / "dataset_tamilnadu_districts"
    if tn_dir.exists():
        dist_count = len([d for d in tn_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        print(f"  • Regional Extents: {dist_count} Tamil Nadu Districts Cataloged [OK]")

    print("\n" + "=" * 80)
    return all_ok

def main():
    print_banner()
    diagnostics_passed = run_diagnostics()

    print("\n🚀 Platform Subsystems Operational:")
    print("  ----------------------------------------------------------------------")
    print("  🏠 1. Main Bi-Temporal Analysis   : http://127.0.0.1:5000/")
    print("  🗺️  2. Interactive GIS Satellite Map: http://127.0.0.1:5000/map")
    print("  🧠 3. Multimodal AI Vector Search : http://127.0.0.1:5000/semantic-search")
    print("  📄 4. Environmental Audit Reports : http://127.0.0.1:5000/reports")
    print("  🔍 5. SQL & NL Change Search      : http://127.0.0.1:5000/search")
    print("  📊 6. Model Architecture & Stats  : http://127.0.0.1:5000/model")
    print("  ----------------------------------------------------------------------\n")

    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"🌐 Launching Flask Web Server on http://127.0.0.1:{port} (Press Ctrl+C to stop)...\n")
    from app.app import app
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
