# 🛰️ System Setup & Deployment Guide: Sentinal

A comprehensive guide for setting up and running the **Semantic Satellite Change Detection, Multimodal Vector Retrieval & Environmental Audit Platform** on a fresh Linux / macOS / Windows system.

---

## 1. System Requirements

- **Operating System:** Linux (Ubuntu 20.04+ / Debian 11+ recommended), macOS, or Windows 10/11 (WSL2 recommended)
- **Python Version:** Python `3.10`, `3.11`, or `3.12`
- **Memory (RAM):** Minimum 8 GB (16 GB recommended for whole-state batch processing)
- **Disk Space:**
  - Codebase & ML Models: ~100 MB
  - Tamil Nadu District Satellite Tiles (optional full dataset): ~14 GB
- **Hardware Acceleration:** 100% CPU Compatible (Optional NVIDIA GPU for faster CLIP embedding extraction)

---

## 2. Quick Start Installation

### Step 1: Clone the Repository
```bash
git clone git@github.com:Premkumar499/sentinal.git
cd sentinal
```

### Step 2: Create and Activate a Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (cmd):
# .venv\Scripts\activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip and Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional for AI Chatbot)
The platform includes an AI Intelligence Assistant powered by Groq LPU inference. To enable live LLM synthesis:
```bash
# Set your Groq API Key (get one free at https://console.groq.com)
export GROQ_API_KEY="your_groq_api_key_here"
```
*(Note: If no API key is provided, the platform automatically uses grounded rule-based telemetry calculations).*

---

## 3. Directory Layout & Architecture

```
sentinal/
├── app/                        # Flask Web Application & Routes
│   ├── app.py                  # Main Flask Server & REST Endpoints
│   ├── templates/              # Jinja2 HTML Templates (Dashboard, GIS Map, Chatbot, Features)
│   └── static/                 # Stylesheets, JS modules, Leaflet assets
├── config/                     # Configuration Profiles
│   ├── app_config.yaml         # Application-level thresholds & database paths
│   └── districts.json          # 38 Tamil Nadu Districts GPS centers & bounds
├── models/                     # Trained Machine Learning Models
│   ├── xgboost_change_detector.json        # 32-feature XGBoost Model (Threshold = 0.72)
│   └── xgboost_change_detector_tuned.json  # Tuned High-Precision Variant
├── src/                        # Core Engine Modules
│   ├── inference.py            # 32-feature extractor & connected component analyzer
│   ├── pipeline.py             # End-to-end change analysis pipeline
│   ├── chatbot_engine.py       # Grounded Multi-temporal Info & Image Chatbot
│   ├── embeddings.py           # 512-d CLIP dual multimodal encoder
│   ├── vector_store.py         # FAISS vector database manager
│   ├── search_engine.py        # SQLite / Geo-index manager
│   ├── temporal_fetcher.py     # Multi-temporal Slippy Tile Downloader
│   └── reporting.py            # Environmental audit & PDF report generator
├── scripts/                    # Automation & Execution Workflows
│   ├── run_full_state_pipeline.py      # Master full-state pipeline runner
│   ├── download_2020_2025.py           # Multi-threaded Wayback tile downloader
│   ├── run_tamilnadu_change_detection.py # Bi-temporal change detection batcher
│   ├── index_satellite_embeddings.py   # FAISS embedding index builder
│   └── build_place_index.py            # Spatial place name indexer
├── run_flask.py                # Web server entrypoint
├── requirements.txt            # Python dependencies
├── README.md                   # Project Overview & Architecture
└── SETUP.md                    # Setup & Deployment Instructions
```

---

## 4. Running the Application

### Option A: Launch the Web Dashboard & Chatbot Directly
To run the web interface with built-in demonstration models:
```bash
python run_flask.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

Key Navigation Routes:
- **Interactive GIS Map:** `http://127.0.0.1:5000/`
- **Info Chat Bot (Intelligence):** `http://127.0.0.1:5000/info-chatbot`
- **Image Chat Bot (Visual Differences):** `http://127.0.0.1:5000/image-chatbot`
- **Three Core AI Features Hub:** `http://127.0.0.1:5000/features`
- **Official Audit PDF Reports:** `http://127.0.0.1:5000/reports`

---

## 5. Automated Data Ingestion & Full Pipeline Execution

To download new satellite imagery (e.g. 2020 vs 2025 orthorectified imagery across districts) and run complete change detection + vector indexing:

### 1. Download Bi-Temporal Satellite Tiles:
```bash
python scripts/download_2020_2025.py --districts karur coimbatore --max-tiles 15 --workers 8
```

### 2. Run the Full-State Pipeline in One Command:
```bash
python scripts/run_full_state_pipeline.py --districts karur --max-tiles 15 --threshold 0.65
```
This executes:
1. Multi-threaded Orthorectified Tile Download
2. Bi-Temporal Alignment & Quality Validation
3. 32-Feature XGBoost Inference Engine
4. Connected Component Region Extraction & GPS Georeferencing
5. FAISS Multimodal Vector Embeddings Indexing
6. Spatial Place Name Index Sync

---

## 6. Running Tests & Verification

Verify that all algorithms, difference percentages, and endpoints pass unit tests:
```bash
pytest tests/ -v
```

---

## 7. Production Deployment (Gunicorn / Systemd)

For production deployment on a cloud server (AWS, Azure, DigitalOcean):
```bash
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 "app.app:app"
```
