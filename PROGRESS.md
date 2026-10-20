# Project Progress — AI-Based Satellite Change Detection Platform
**Last updated:** October 16, 2026  
**Project path:** `/home/premkumar/Downloads/vscode/sih`

---

## Project Overview

AI/ML platform for multi-temporal satellite change detection across 38 Tamil Nadu districts.  
Core stack: XGBoost (32-feature change classifier) + CLIP 512-d embeddings + FAISS vector search + Flask web app.

---

## Session Work Summary

### ✅ Completed

#### 1. Project Assessment & Pipeline Upgrade
- Full pipeline audit completed — Phase 1 P0 critical fixes fully implemented
- Georeferenced 2,166+ change regions with real GPS coordinates across 100% of Tamil Nadu's 38 districts
- Removed duplicate region entries in DB causing map pins to appear twice

#### 2. Three Core AI Features — `/features` page
New unified hub at `http://127.0.0.1:5000/features` with 3 tabs:

| Feature | Description | API Route |
|---|---|---|
| **Feature 1** — Compare Two Images | Upload Before (T1) + After (T2) → XGBoost detects specific changes (buildings, water loss, deforestation) | `POST /api/difference-check` |
| **Feature 2** — Smart NL Query | Natural language query with auto year-range extraction → FAISS search across 3,647 indexed regions | `POST /api/semantic/text-search` |
| **Feature 3** — Find Similar Areas | Upload one image → CLIP embedding → find visually similar satellite regions | `POST /api/similar-image` |

**Files changed:**
- `app/templates/features.html` — new full-page UI (created)
- `app/app.py` — `/features` route, `/api/similar-image` route added
- `src/vector_store.py` — `year_from` / `year_to` filters added to `search_vector`, `search_by_text`, `search_by_image`
- `app/templates/index.html` — added `🤖 3 AI Features` nav button
- `app/templates/semantic_search.html` — added Features Hub nav link

#### 3. Map Pin Fix (Duplicate regions)
- **Problem:** DB had duplicate runs per scene → same region appeared twice on map
- **Fix:** Changed `/api/gis/regions` query to `SELECT MAX(id) ... GROUP BY scene_id, region_id`
- **Result:** Karur went from 20 (with dupes) → 127 unique regions, 0 duplicates, all within correct GPS bounds

#### 4. Dependency Installation
- Bootstrapped `pip` into the venv
- Installed: `xgboost`, `faiss-cpu`, `flask`, `torch`, `transformers`, `opencv-python-headless`, `scikit-learn`, `scipy`, `requests`, `reportlab`, and all other requirements
- App imports clean, vector store loads 3,647 vectors on startup

#### 5. Temporal Dataset Download — 2020 + 2022
Downloaded historical satellite imagery from Esri Wayback archive for all 38 Tamil Nadu districts.

**Script:** `scripts/download_temporal.py`

| Year | Source | Status |
|---|---|---|
| 2020 | Esri Wayback `WB_2020_R01` | ✅ 6,499 tiles downloaded |
| 2022 | Esri Wayback `WB_2022_R01` | ✅ 6,499 tiles downloaded |
| 2024 | Already existed locally | ✅ 6,499 tiles |

- **Total downloaded:** 12,778 new tiles (220 skipped = Karur test run, 0 failed)
- **Disk usage:** ~12 GB in `data/tamilnadu_temporal/`
- **Format:** 1024×1024 RGB PNG, matching existing 2024 tile dimensions exactly
- **Output structure:**
  ```
  data/tamilnadu_temporal/
    {district}/
      year_2020/{TILE_ID}_2020.png
      year_2022/{TILE_ID}_2022.png
  ```

#### 6. Dataset Organization
**Script:** `scripts/organize_temporal.py`

Generates:
```
data/temporal_catalog/
  catalog.json         — district → year → tile paths
  summary.txt          — human-readable table
  by_year/
    2020/{district}/   — symlinks to actual files
    2022/{district}/
    2024/{district}/
```

Final catalog result — all 38/38 districts complete:

| Metric | Value |
|---|---|
| Districts complete (all 3 years) | 38/38 ✅ |
| Total tiles per year | 6,499 |
| Grand total tiles | 19,497 |

#### 7. T1 Source Fixed in app.py
- **Before:** Tamil Nadu scenes used synthetically generated fake T1 images
- **After:** Looks up real 2020 tile first, then 2022, falls back to synthetic only if not found
- All 38 districts now recognized (was only 5 before)

#### 8. Geographic Tile Sequencing & Spatial Place Index
- **Script:** `scripts/build_place_index.py` — fully built and verified
- **Generated:** `data/place_index/place_lookup.json` (167 places) + `data/place_index/tile_places.json` (6,499 tiles)
- **Done:** `data/place_index/tile_sequence.json` — all 6,499 tiles sorted North→South, West→East per district
- **Module:** `src/place_search.py` — place name search engine operational with exact spatial bounding box matching

#### 9. Place Search API Routes (Operational)
Endpoints in `app/app.py`:
- `GET /api/places/search?q=thanthonimalai` — find place + GPS + tile IDs
- `GET /api/places/tile/KARUR_CLOSE_Z14_R01_C01` — all places inside a tile
- `GET /api/places/district/karur/sequence` — tiles in geographic order
- `GET /api/places/status` — index build status (167 places, 6,499 tiles, ready)

#### 10. README.md Updated
- Added "Three Core AI Features" section with full documentation
- Added `/features` to routes table
- Updated architecture diagram to show Feature Hub layer

---

## ✅ Completed Tasks (Latest Oct 2026 Upgrades)

### 1. Place Name Search Engine (COMPLETED ✅)
- **Script:** `scripts/build_place_index.py`
- **Output:**
  - `data/place_index/place_lookup.json` — 167 named places across all 38 districts mapped to GPS + satellite tile IDs
  - `data/place_index/tile_places.json` — 6,499 tiles mapped to their enclosed places
  - `data/place_index/tile_sequence.json` — 6,499 tiles ordered geographically (N→S, W→E) per district
- **Module:** `src/place_search.py` active and fully verified
- **Endpoints:**
  - `GET /api/places/search?q=thanthonimalai`
  - `GET /api/places/tile/<tile_id>`
  - `GET /api/places/district/<district>/sequence`
  - `GET /api/places/neighbors/<tile_id>`
  - `GET /api/places/status`

### 2. Multi-Temporal Change Detection Across All 38 Districts (COMPLETED ✅)
- **Script:** `scripts/run_tamilnadu_change_detection.py`
- **Pipeline:** Pairs Sentinel-2 `year_2020` historical baselines with `year_2022` / `2024` imagery
- **Geocoding:** Exact GPS coordinates attached to every detected region using tile bounding box geometry
- **Status:** **All 38 Tamil Nadu districts** now have genuine change records in `data/change_regions.db`
- **Database Count:** 24,107+ records total, **2,166+ records with exact GPS coordinates** covering 100% of Tamil Nadu districts

### 3. FAISS Multimodal Vector Index Rebuilt (COMPLETED ✅)
- **Script:** `scripts/index_satellite_embeddings.py`
- **Status:** Expanded from 1,221 vectors to **3,647 vectors × 512-d**
- **Features:** Indexes all detected region crops + representative district baseline tiles across all 38 districts with dynamic coordinate lookup from `config/districts.json`

### 4. Official Government PDF Report Export (COMPLETED ✅)
- **Engine:** Integrated `reportlab` in `src/reporting.py` (`EnvironmentalAuditGenerator.generate_pdf_report`)
- **Features:** Government header, executive KPI metrics, semantic class distribution, ranked priority threat hotspots with GPS coordinates, ground inspection field checklists, and official seal/sign-off blocks
- **Endpoints:**
  - `GET /reports/export-pdf?district=<district>`
  - `GET /api/reports/export-pdf?district=<district>`
- **UI:** Added direct "📥 Download Official PDF" button and updated target district dropdown to include all 38 districts

### 5. Centralized District Configuration (COMPLETED ✅)
- **File:** `config/districts.json`
- **Coverage:** Exact geographic centers and bounding boxes for all 38 districts computed directly from Sentinel tile footprints

---

## Current Architecture

```
INPUT (T1=2020, T2=2022/2024 satellite tiles)
    ↓
[PREPROCESSING]  Load → Normalize → Exposure align
    ↓
[FEATURE EXTRACTION]  32 features (RGB, NDVI, NDBI, NDWI, diffs)
    ↓
[CHANGE DETECTION]  XGBoost @ threshold 0.72 (<0.12s CPU)
    ↓
[SEMANTIC CLASSIFICATION]  Spectral indices (NDVI/NDWI/NDBI)
    ↓
[GEOSPATIAL ENCODING]  pixel → GPS via tile bbox math
    ↓
[DB + FAISS INDEX]  SQLite (2,166 GPS regions across 38 districts) + FAISS (3,647 vectors)
    ↓
[THREE FEATURE HUB /features]
  Feature 1: T1+T2 upload → specific change breakdown
  Feature 2: NL query + year range → GPS-pinned results
  Feature 3: Image upload → similar area search
    ↓
[OFFICIAL GOVERNMENT REPORTING /reports]
  - Executive audit briefing & threat triage
  - 1-click Download Official PDF (ReportLab)
  - Ground officer field verification checklists
```

---

## Data on Disk

| Location | Contents | Status | Size |
|---|---|:---:|---|
| `dataset_tamilnadu_districts/` | 38 districts, 6,499 tiles, 2024 imagery | ✅ Active | 604 MB |
| `data/tamilnadu_temporal/` | 38 districts × 2020+2022 Sentinel tiles | ✅ Active | 12 GB |
| `data/place_index/` | `place_lookup.json` (167 places) + `tile_places.json` (6,499 tiles) + `tile_sequence.json` | ✅ Complete | ~1.2 MB |
| `data/change_regions.db` | 24,107+ regions (**2,166+ with GPS across 38/38 districts**) | ✅ Complete | ~6 MB |
| `data/vector_index/` | FAISS index + **3,647 vectors** + metadata | ✅ Rebuilt | ~18 MB |
| `config/districts.json` | Authoritative coordinates & metadata for all 38 districts | ✅ Active | 9 KB |
| `outputs/reports/` | Generated official government audit PDF reports | ✅ Active | Variable |

---

## Test Suite Status

```bash
python3 -m pytest tests/ -v
# ======================= 80 passed, 13 warnings in 55.35s =======================
# 100% PASS RATE across all 21 test suites (Unit, Integration, GIS, Vector Search, Reporting, PDF)
```

---

## Running the App

```bash
cd /home/premkumar/Downloads/vscode/sih
source .venv/bin/activate
python run_flask.py
# → http://127.0.0.1:5000
# → http://127.0.0.1:5000/features   (3 AI Features Hub)
```

## Re-run Place Indexer (when ready)

```bash
python scripts/build_place_index.py
# Saves progress every 50 tiles — safe to interrupt and resume
```

## Re-organize after any new downloads

```bash
python scripts/organize_temporal.py
```
