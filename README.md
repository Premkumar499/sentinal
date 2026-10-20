# 🛰️ Semantic Satellite Change Detection, Multimodal AI Search & Environmental Audit Platform

**AI/ML-Based Multi-Temporal Satellite Image Change Detection, Multimodal Vector Retrieval & Environmental Audit Intelligence**

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![ML Core](https://img.shields.io/badge/ML%20Core-XGBoost%20%2B%20FAISS-orange)](https://xgboost.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Hardware-100%25%20CPU%20Compatible-brightgreen)](#)
[![Validation Status](https://img.shields.io/badge/Validation-80%2F80%20Tests%20Passed-success)](#)
[![Release Version](https://img.shields.io/badge/Release-v1.2.0--final-blueviolet)](#)

---

## 1. Executive Overview & Problem Statement

Given multi-temporal satellite imagery acquired at two distinct timestamps ($T_1$ and $T_2$), detecting land-surface changes, understanding their semantic nature, and taking administrative action is vital for disaster response, urban planning, and ecological conservation.

This platform bridges raw pixel change detection with high-level geospatial intelligence:
1. **Physical Surface Change Detection:** Tuned 32-feature **XGBoost classifier** (decision threshold = 0.72) detecting physical changes in $< 0.12$ seconds on CPU.
2. **Real-World Regional Ingestion:** Integrated Sentinel-2 and Esri Wayback imagery across **all 38 Tamil Nadu districts**, georeferencing detected regions with true GPS coordinates ($\text{Lat}, \text{Lon}$) and ground area in $m^2$ (2,166+ geocoded regions).
3. **Multimodal AI Semantic Vector Search:** Dual-encoder embedding engine (512-d CLIP & FAISS) enabling **Text $\to$ Satellite Area Search** (*'Find large water bodies near urban regions'*) and **Image $\to$ Similar Area Search** across **3,647 indexed satellite regions and tiles**.
4. **Interactive GIS Satellite Map:** Leaflet.js map with **Esri World Imagery (Satellite)** and **CartoDB Dark Matter** base-layers, dynamic GPS change markers scaled by area, and 4-panel visual comparison popups covering all 38 districts via `config/districts.json`.
5. **Government Environmental Audit & Threat Reporting:** Automated threat triage (lakebed encroachment alerts, deforestation flags), executive KPI briefings, and 1-click **Official Downloadable & Printable ReportLab PDF Reports** (`/reports/export-pdf`) with field verification checklists for ground officers.
6. **Three Core AI Feature Hub (`/features`):** Unified interface exposing — (1) bi-temporal image comparison with semantic class breakdown, (2) natural language location query with year-range filtering over 3,647 FAISS-indexed regions, and (3) single-image visual similarity search using 512-d CLIP embeddings.
7. **Place Name Search Engine (`/api/places/*`):** Spatial place name index mapping 167 towns, taluks, and cities to 6,499 satellite tiles for instantaneous query resolution (e.g. *Thanthonimalai*, *Gandhigramam*, *Ooty*).

---

## 2. System Architecture

```
                                  SATELLITE IMAGERY
                       (Sentinel-2 / 38 Tamil Nadu Districts)
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  PHYSICAL ENGINE                MULTIMODAL SEARCH                   GIS ENGINE
 (32-Feature XGBoost            (Dual 512-d CLIP &              (Leaflet.js + Esri
  Change Classifier)              FAISS Vector Store)           Satellite + Footprints)
        │                                │                                │
        ▼                                ▼                                ▼
  Change Contours &           Text/Image-to-Satellite          Interactive GPS Map
  Pixel Probability           Area Search + Year Filter        with Area-Scaled
  (2,166+ GPS Regions)        (3,647 Vectors, FAISS IP)        Pulsing Markers (38 Districts)
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         ▼
                          THREE CORE AI FEATURE HUB (/features)
                   ┌─────────────────────────────────────────────┐
                   │  Feature 1: Upload T1+T2 → Specific Changes  │
                   │  Feature 2: NL Query + Year Range → GPS Pins │
                   │  Feature 3: Upload Image → Similar Areas      │
                   └─────────────────────────────────────────────┘
                                         │
                                         ▼
                     GOVERNMENT ENVIRONMENTAL AUDIT ENGINE
                            (src/reporting.py)
            - Automated Threat Triage (Critical Lakebed Encroachments)
            - Hectares & Class Transition Depletion Breakdown
            - Actionable Ground Verification Checklist
                                         │
                                         ▼
                     OFFICIAL BRIEFING DASHBOARD (/reports)
                (1-Click Download Official ReportLab PDF Export)
```

---

## 3. Three Core AI Features

The platform exposes three primary user-facing AI capabilities at `/features`:

### Feature 1 — Compare Two Satellite Images (Specific Change Detection)
Upload a **Before (T1)** and **After (T2)** image. The XGBoost classifier detects only semantically meaningful changes — new buildings, water body loss, deforestation, road expansion — not noise or lighting shifts. Returns:
- Changed area percentage and pixel counts
- Semantic class breakdown (water / building / road / vegetation / bare land)
- Top land-cover transitions (e.g. vegetation → bare land)
- Full 6-panel visual analysis at `/results/<run_id>`

**Route:** `POST /api/difference-check` (form: `file_t1`, `file_t2`)

### Feature 2 — Smart Natural Language Location Query with Time Range
Type a plain-language question like:
> *"Find locations in Karur where new buildings appeared between 2025 and 2026"*

The system:
1. Auto-extracts district, semantic class, and year range from the query
2. Encodes it into a 512-d CLIP embedding
3. Searches 3,647 indexed satellite regions via FAISS cosine similarity
4. Filters results by extracted year range (`year_from` / `year_to`)
5. Returns GPS-pinpointed matching regions with map links

**Route:** `POST /api/semantic/text-search` (JSON: `query`, `year_from`, `year_to`, `district`, `semantic_class`)

### Feature 3 — Upload One Image → Find Visually Similar Areas
Upload any satellite image crop. The CLIP encoder converts it to a 512-d vector and FAISS retrieves the most visually similar indexed regions across all 38 Tamil Nadu districts — matching spectral texture, color distribution, and structural patterns.

**Route:** `POST /api/similar-image` (form: `image`, `district`, `top_k`)

**Unified Hub URL:** `http://127.0.0.1:5000/features`

---
## 4. Web Subsystems & Live URLs

| Subsystem | Live Route | Target Audience | Key Feature |
| :--- | :--- | :--- | :--- |
| **🤖 AI Feature Hub** | `http://127.0.0.1:5000/features` | All Users | 3-in-1: Compare Images · Smart Query · Find Similar Areas |
| **🏠 Change Detection Dashboard** | `http://127.0.0.1:5000/` | Remote Sensing Analysts | 6-Panel bi-temporal comparison ($T_1$, $T_2$, Mask, Probability, Semantic Map, Overlay) |
| **🗺️ Interactive GIS Satellite Map** | `http://127.0.0.1:5000/map` | GIS Specialists & Planners | Esri Satellite base-layer, GPS change markers scaled by area ($m^2$), 38-district flight shortcuts |
| **🧠 Multimodal AI Vector Search** | `http://127.0.0.1:5000/semantic-search` | Search & Intelligence Teams | Plain-text to satellite area search & image-to-similar-area retrieval using 512-d FAISS index (3,647 vectors) |
| **📄 Environmental Audit Reports** | `http://127.0.0.1:5000/reports` | District Collectors & TWAD | Automated threat triage, lakebed encroachment alerts, 1-click downloadable official ReportLab PDF (`/reports/export-pdf`) |
| **🔍 SQL & NL Query Engine** | `http://127.0.0.1:5000/search` | Database Analysts | Structured SQL queries, area thresholds, and transition filtering |
| **📊 Model Architecture & Audit** | `http://127.0.0.1:5000/model` | ML Evaluators | 32-feature global importance rankings and frozen threshold verification |

---

## 4. Scientific Validation Results

Evaluated across **20 unseen test scenes** from the OSCD dataset under strict zero-tuning protocols:

| Metric | XGBoost Tuned | Simple Difference Baseline | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Mean F1-Score** | **0.4606 ± 0.2125** | 0.2184 | **+110.9%** |
| **Mean IoU** | **0.3244 ± 0.1835** | 0.1265 | **+156.4%** |
| **Mean Precision** | **0.5621 ± 0.1699** | 0.2840 | **+97.9%** |
| **Mean Recall** | **0.4635 ± 0.2833** | 0.3120 | **+48.6%** |
| **Inference Latency**| **< 0.12 seconds** | 0.05 seconds | **Real-Time CPU** |

---

## 5. Installation & Quick Start

### Step 1: Environment Setup
```bash
# Clone repository and enter directory
cd /home/premkumar/Downloads/vscode/sih

# Install dependencies
pip install -r requirements.txt
```

### Step 2: One-Click System Launch
```bash
python run_project.py
```
This runs pre-flight diagnostics and starts the Flask server at `http://127.0.0.1:5000`.

### Step 3: Run Automated Test Suite
```bash
python3 -m pytest tests/ -v
```
*(All 80/80 tests pass cleanly across unit, integration, GIS, vector search, reporting, and PDF export suites).*

---

- **System Setup & Deployment Guide:** [`SETUP.md`](file:///home/premkumar/Downloads/vscode/sih/SETUP.md)
- **Presentation & Viva Guide:** [`docs/SIH_FINAL_PRESENTATION_GUIDE.md`](file:///home/premkumar/Downloads/vscode/sih/docs/SIH_FINAL_PRESENTATION_GUIDE.md)
- **Scientific Audit Report:** [`docs/FINAL_PROJECT_REPORT.md`](file:///home/premkumar/Downloads/vscode/sih/docs/FINAL_PROJECT_REPORT.md)
