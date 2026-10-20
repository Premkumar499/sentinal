# Semantic Satellite Change Detection System — Project Manifest

**Project Name:** Semantic Satellite Change Detection  
**Primary Dataset:** OSCD (Onera Satellite Change Detection)  
**Machine Learning Model:** XGBoost Tuned Classifier (Threshold = 0.72)  
**Framework:** Flask Web Application & Unified Master Pipeline  
**Hardware Requirement:** 100% CPU Compatible  

---

## Repository Structure & Core Modules

```
sih/
├── app/                        # Flask Web Application
│   ├── app.py                  # Main Web Server & REST API Endpoints
│   ├── templates/              # HTML Templates (index, result, search, map, region_detail, model_info, 400, 404, 500)
│   └── static/                 # CSS, JavaScript & Static Assets
├── config/                     # Configuration Locks
│   ├── final_release.yaml      # System Release Metadata Lock
│   ├── app_config.yaml         # Centralized System Settings & Paths
│   ├── day13_config.yaml       # Explainability Settings
│   └── demo_scenes.json        # Benchmark Scene Definitions
├── data/                       # Dataset & Region Database
│   ├── oscd/                   # OSCD Satellite Imagery Dataset
│   └── change_regions.db       # SQLite Change Region Database
├── docs/                       # Technical Documentation & Reports
│   ├── DAY14_REPORT.md         # Final Integration Report
│   ├── DAY14_MODULE_AUDIT.md   # System Module Audit Table
│   ├── DAY14_TEST_REPORT.md    # Test & Verification Report
│   ├── PERFORMANCE.md          # Latency & Profiling Report
│   └── PROJECT_MANIFEST.md     # Project Manifest File
├── models/                     # Trained ML Model Weights
│   └── xgboost_change_detector.json
├── outputs/                    # Output Artifacts & Runs
│   ├── presentation/           # Presentation Slide Visuals (day14_final_system.png, etc.)
│   ├── runs/                   # Isolated Session Run Directories
│   └── validation/             # Reproducibility & Timing CSV Artifacts
├── scripts/                    # Release & Utility Scripts
│   ├── check_database.py       # Database Integrity Auditor
│   ├── run_demo.sh             # Demo Launch Runner
│   └── run_final_evaluation.py # Final Evaluation Script
├── src/                        # Core Python Engine Subsystems
│   ├── pipeline.py             # Master Pipeline Execution Entry Point
│   ├── inference.py            # End-to-End Inference Engine
│   ├── preprocessing/          # Data Loading & Normalization
│   ├── features/               # 32 Spectral & Spatial Feature Extraction
│   ├── models/                 # XGBoost Classifier Implementations
│   ├── semantic_change.py      # Rule-Based Land-Cover Classification
│   ├── region_analysis.py      # Connected Component Region Extraction
│   ├── search_engine.py        # SQLite Database Search Engine
│   ├── geospatial.py           # Rasterio Coordinates & GeoJSON Engine
│   └── explanation.py          # Deterministic Region Explanation Engine
└── tests/                      # Automated System Test Suite (60 Test Cases)
```
