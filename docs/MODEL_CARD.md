# Model Card: Tuned XGBoost Satellite Change Detector

## 1. Model Details
- **Model Architecture:** XGBoost (Extreme Gradient Boosting Binary Classifier)
- **Model Version:** 1.0.0 (Tuned Post-Processing Parameters)
- **Model Weights File:** `models/xgboost_change_detector_tuned.json`
- **Config File:** `models/reproducible_config.yaml`
- **Task:** Binary change detection in multi-temporal Sentinel-2 imagery
- **Hardware Requirement:** CPU Compatible (Zero GPU requirement)

---

## 2. Model Input & Output
- **Inputs:** Pair of bi-temporal satellite image bands ($T_1$ and $T_2$) scaled to normalized reflectance $[0.0, 1.0]$.
- **Features (32 Engineered Features):**
  - Raw spectral bands ($T_1, T_2$): Red, Green, Blue, NIR, SWIR
  - Absolute Band Differences ($|T_2 - T_1|$)
  - Band Ratios & Normalized Difference Indices (NDVI, NDWI, NDBI)
  - Index Differences ($\Delta\text{NDVI}, \Delta\text{NDWI}, \Delta\text{NDBI}$)
- **Output:** Continuous change probability map $[0.0, 1.0]$ and binarized change mask ($0 = \text{unchanged}, 1 = \text{changed}$) at threshold $0.72$.

---

## 3. Training & Validation Strategy
- **Dataset:** ONERA Satellite Change Detection (OSCD)
- **Split:** 14 Training Scenes / 10 Test Scenes
- **Imbalance Mitigation:** `scale_pos_weight` tuned to account for minority change class (<15% pixels)
- **Threshold Calibration:** Optimal threshold identified as $0.72$ via F1/Precision/Recall trade-off optimization.

---

## 4. Performance Metrics (Averaged across Test Scenes)
- **Precision:** 0.5336
- **Recall:** 0.5366
- **F1 Score:** 0.5351
- **IoU (Jaccard Index):** 0.3653
- **Average Inference Latency:** `0.0928 seconds` on CPU

---

## 5. Semantic Layer Qualification
- **Semantic Source:** Spectral rules (NDVI/NDWI/NDBI) + morphological thinness heuristics (Distance Transform).
- **Disclaimer:** The OSCD dataset provides binary change masks only. Semantic labels indicate *"change overlapping [class]"* and should not be asserted as confirmed ground truth construction without auxiliary field verification.
