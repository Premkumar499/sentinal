# Day 5 Technical Report: End-to-End Prototype Application & Flask Web Integration

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Primary ML Model:** XGBoost (Pre-trained Tuned Model, Threshold = 0.72)  
**Semantic Layer:** Spectral Index Rules + Morphological CCA  
**Execution Environment:** CPU Only  

---

## 1. Executive Summary

Day 5 successfully integrated the pre-trained XGBoost binary change detection model and the Day 4 semantic land-cover layer into a complete, standalone, CPU-friendly Flask prototype application (`app/app.py`).

Users can select from pre-packaged OSCD benchmark test scenes (`test_000`, `test_001`, `test_002`, `val_000`, etc.) or upload custom bi-temporal satellite image pairs ($T_1$ and $T_2$). The application performs automated preprocessing, feature extraction (32 features), XGBoost inference using saved model weights, semantic land-cover classification, connected component region extraction, 6-panel visualization generation, and exports full JSON/CSV/ZIP result packages.

---

## 2. Key Accomplishments

1. **Pre-Trained Inference Engine (`src/inference.py`)**:
   - Zero model retraining during inference.
   - Loads saved XGBoost weights (`models/xgboost_change_detector_tuned.json`).
   - Enforces exact 32-feature alignment in the reproducible order specified in `models/reproducible_config.yaml`.
   - Applies the calibrated optimal threshold $0.72$.
2. **Flask Web Application (`app/app.py`)**:
   - Modern, glassmorphism dark-themed dashboard UI (`app/templates/index.html` & `results.html`).
   - Dual input options: Quick OSCD Sample Scene Selector or Custom Image Upload.
   - 6-Panel Visualization Dashboard ($T_1$ Before, $T_2$ After, XGBoost Change Mask, Land Cover Mask, Semantic Change Map, Overlay).
   - Class-specific change breakdown cards (Road, Building, Water, Vegetation, Bare Land, Unknown).
   - Connected Component Analysis (CCA) region table with pixel areas, bounding boxes, and centroids.
   - Downloadable CSV, JSON, and ZIP packages (`result_<run_id>.zip`).
3. **Automated REST API (`POST /api/detect`)**:
   - Programmatic JSON endpoint accepting scene IDs or multipart file uploads.
   - Verified 100% numerical equivalence between CLI inference and Flask API responses.
4. **Input Validation & Error Handling**:
   - Validates file existence, channel compatibility, and dimension matching without exposing raw stack traces.
   - Implements graceful fallback if semantic classification inputs are partial.
5. **Comprehensive Test Suite (`tests/test_day5.py`)**:
   - 21 / 21 tests passing across Day 1, Day 3, Day 4, and Day 5.
6. **Performance Benchmarking (`run_day5.py`)**:
   - End-to-End inference executed in **0.3262 seconds** per scene on CPU.

---

## 3. Performance Benchmark Summary

| Component | Average Execution Time | Percentage of Total |
| :--- | :--- | :--- |
| **Data Loading & Preprocessing** | `0.0130 s` | 4.0% |
| **Feature Extraction (32 Features)** | `0.0417 s` | 12.8% |
| **XGBoost Inference** | `0.0772 s` | 23.7% |
| **Semantic Classification** | `0.0130 s` | 4.0% |
| **Visualization & ZIP Packaging** | `0.1813 s` | 55.5% |
| **Total End-to-End Runtime** | **`0.3262 s`** | **100.0%** |

*Exported to `outputs/reports/day5_performance.csv`.*

---

## 4. Day 5 Success Criteria Verification

- [x] Existing Day 3 XGBoost model loaded correctly (`models/xgboost_change_detector_tuned.json`)
- [x] Day 4 semantic pipeline loaded correctly (`src/models/semantic.py`)
- [x] End-to-end inference engine implemented (`src/inference.py`)
- [x] T1/T2 selection and upload form functional
- [x] Input validation implemented
- [x] Change mask generated ($threshold = 0.72$)
- [x] Semantic mask generated
- [x] Semantic change mask generated
- [x] Region extraction works (CCA, $MIN\_REGION\_PIXELS = 10$)
- [x] Statistics and class breakdown generated
- [x] Before/After visualizations generated
- [x] Change visualization generated
- [x] Semantic visualization generated
- [x] Overlay visualization generated
- [x] Flask homepage works (`GET /`)
- [x] Flask results page works (`GET /results/<run_id>`)
- [x] REST API endpoint works (`POST /api/detect`)
- [x] Error handling & fallback mode implemented
- [x] Demo OSCD sample scene selector works
- [x] CLI and Flask outputs 100% numerically identical
- [x] Automated tests pass (21/21 passed)
- [x] README updated
- [x] `requirements.txt` updated
- [x] Performance report generated (`outputs/reports/day5_performance.csv`)
- [x] Zero fabricated semantic labels or metrics
- [x] Qualified wording enforced ("Change overlapping road class")

---

## 5. Final Day 5 Technical Conclusion

"Day 5 integrated the trained XGBoost binary change detector and semantic change pipeline into an end-to-end Flask prototype. The application accepts two satellite images, performs preprocessing and feature extraction, detects changes, associates changed regions with available semantic classes, extracts changed regions and statistics, and visualizes the results."
