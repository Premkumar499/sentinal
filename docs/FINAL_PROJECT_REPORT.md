# Final Project Technical Report

**Project Title:** AI/ML-Based Semantic Change Detection System Using Multi-Temporal Satellite Images  
**Short Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Primary Change Model:** XGBoost Tuned Classifier (Threshold = 0.72)  
**Semantic Engine:** Spectral Index Rules (NDVI/NDWI/NDBI) + Distance Transform Heuristics  
**Deployment Stack:** Flask Web Application + REST API (100% CPU Compatible)  
**Version:** v1.0.0-final  

---

## 1. Title & Metadata
- **Project Name:** Semantic Satellite Change Detection System
- **Authors & Team:** Project Development Team
- **Target Platform:** Linux/Unix (100% CPU Compatible, zero GPU requirement)
- **Status:** Final Release (Day 10 Verified)

---

## 2. Abstract
Given multi-temporal satellite imagery acquired at two distinct dates ($T_1$ and $T_2$), detecting land-surface changes and interpreting their semantic categories is vital for urban planning, environmental monitoring, and infrastructure management. This project presents a CPU-friendly, end-to-end machine learning system trained on the OSCD (Onera Satellite Change Detection) dataset. The pipeline extracts 32 tabular spectral, temporal, and spatial features and uses a tuned XGBoost classifier operating at probability threshold $0.72$ to generate binary change masks. A spectral reference layer (NDVI, NDWI, NDBI) intersects detected changes to assign land-cover categories (Roads, Buildings, Water, Vegetation, Bare Land, Unknown). Object-level connected components are extracted, ranked by Significance Score, and served via an interactive Flask web UI and REST API. On 20 unseen OSCD test scenes, the system achieved a Mean F1-score of $0.4606 \pm 0.2125$ and Mean IoU of $0.3244 \pm 0.1835$, outperforming the simple difference baseline by $+110.9\%$ F1 improvement.

---

## 3. Introduction
Satellite remote sensing enables continuous monitoring of Earth's surface. However, raw binary change maps indicate only *where* spectral differences occur, leaving domain analysts to manually inspect scenes to determine the nature of the change. This project bridges pixel-level change detection with object-level semantic interpretability.

---

## 4. Problem Statement
Manual satellite change interpretation across vast geographic extents is slow and resource-intensive. Existing deep learning approaches often require heavy GPU infrastructure unavailable in edge deployment environments. There is a need for a lightweight, CPU-compatible machine learning prototype capable of automated change detection and qualified semantic region interpretation.

---

## 5. Objectives
1. Build a modular data ingestion pipeline for Sentinel-2 multispectral imagery.
2. Engineer 32 spectral, temporal, texture, and spatial features.
3. Train and tune a CPU-friendly XGBoost binary change detector.
4. Implement a spectral rule-based semantic layer for reference class association.
5. Extract bounded regions, calculate Significance Scores, and categorize change types.
6. Provide an interactive Flask web application and programmatic REST API.
7. Conduct rigorous, honest scientific validation across unseen test scenes.

---

## 6. Existing vs Proposed Approach

| Aspect | Traditional Difference Baseline | Proposed XGBoost + Semantic System |
| :--- | :--- | :--- |
| **Method** | Fixed Euclidean Band Thresholding | 32-Feature XGBoost Supervised Learning |
| **Metrics** | F1 = 0.2184, IoU = 0.1265 | **F1 = 0.4606, IoU = 0.3244** |
| **Noise Resilience**| High false positive rate from shadows | Morphological post-processing + threshold 0.72 |
| **Output Type** | Raw binary mask | Object regions, bounding boxes, qualified semantic categories |

---

## 7. Dataset (OSCD)
The ONERA Satellite Change Detection (OSCD) dataset contains 100 aligned Sentinel-2 image pairs covering diverse global urban and natural regions at 10m Ground Sample Distance (GSD). Ground-truth masks provide pixel-level binary change annotations.

---

## 8. Data Preprocessing
- Band selection: Blue (B02), Green (B03), Red (B04), NIR (B08), SWIR (B11).
- Reflectance normalization to range $[0.0, 1.0]$.
- Automated dimension matching and bicubic alignment.

---

## 9. Feature Engineering (32 Features)
1. **Spectral Differences:** $\Delta\text{Red}, \Delta\text{Green}, \Delta\text{Blue}, \Delta\text{NIR}, \Delta\text{SWIR}$
2. **Normalized Index Shifts:** $\Delta\text{NDVI}, \Delta\text{NDBI}$
3. **Reflectance Ratios:** Band $T_2 / (T_1 + \epsilon)$
4. **Local Texture Statistics:** $3 \times 3$ sliding window mean and standard deviation for $T_1$ and $T_2$
5. **Spatial Edge Magnitudes:** Sobel filter gradients ($\text{Mag}_{T1}, \text{Mag}_{T2}, \Delta\text{Mag}$)
6. **Spatial Context:** Distance transform to nearest built-up cluster.

---

## 10. XGBoost Change Model
- **Hyperparameters:** `n_estimators=150`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Class Balancing:** `scale_pos_weight` set to negative/positive pixel ratio.
- **Inference Threshold:** $0.72$ (optimizing validation set F1).

---

## 11. Semantic Reference Layer
- **Water:** $\text{NDWI} > 0.1$ ($\text{NDWI} = (\text{Green} - \text{NIR})/(\text{Green} + \text{NIR})$)
- **Vegetation:** $\text{NDVI} > 0.3$ ($\text{NDVI} = (\text{NIR} - \text{Red})/(\text{NIR} + \text{Red})$)
- **Built-up:** $\text{NDBI} > 0.0$ ($\text{NDBI} = (\text{SWIR} - \text{NIR})/(\text{SWIR} + \text{NIR})$)
  - Road: Morphological thickness $\le 5\text{px}$
  - Building: Morphological thickness $> 5\text{px}$
- **Bare Land:** Low NDVI, NDBI, and NDWI.
- **Unknown:** Unclassified pixels ($C = 6$).

---

## 12. Change Detection Pipeline & Post-Processing
1. Feature extraction $\rightarrow$ XGBoost probability array $P(\text{Change})$.
2. Binary thresholding at $0.72$.
3. Morphological $3 \times 3$ closing.
4. Connected Component Analysis filtering isolated regions $< 3$ pixels.

---

## 13. Object Region Analysis & Significance Ranking
Connected components with area $\ge 10$ pixels are extracted as bounding box objects $(X, Y, W, H)$.
- **Significance Score:** $(\text{Area}/100) \times P_{\text{mean}}$
- **Change Types:** `New`, `Removed`, `Expanded`, `Transition`, `Unknown`.

---

## 14. Flask Application Architecture
- **Web UI:** Modern dark glassmorphism dashboard (`app/templates/index.html`, `results.html`, `region_detail.html`).
- **REST API:** Endpoint `POST /api/detect` returning JSON payload with change statistics, region bounding boxes, and transition matrices.
- **Presentation Route:** `@app.route("/demo")` for 1-click execution without upload dependencies.

---

## 15. Experimental Setup & Protocol
- **Partitioning:** 60 Train scenes, 20 Validation scenes, 20 Test scenes.
- **Freeze Policy:** Zero hyperparameter tuning performed on test set during Day 9 evaluation.

---

## 16. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, IoU (Jaccard Index), Confusion Matrix values ($TP, TN, FP, FN$).

---

## 17. Quantitative Results
- **Test Set Mean F1-Score:** $0.4606 \pm 0.2125$ (Median: $0.4591$)
- **Test Set Mean IoU:** $0.3244 \pm 0.1835$ (Median: $0.3004$)
- **Test Set Mean Precision:** $0.5621 \pm 0.1699$
- **Test Set Mean Recall:** $0.4635 \pm 0.2833$
- **CPU Inference Speed:** $< 0.38$ seconds per scene.

---

## 18. Error Analysis & Category Distribution
- **False Positives:** Seasonal vegetation shifts ($38.2\%$), Illumination shifts ($25.5\%$), Shadows ($16.4\%$).
- **False Negatives:** Resolution limits truncating small objects $< 10\text{m}$ ($41.2\%$), Low spectral contrast ($25.9\%$).

---

## 19. Semantic Results & Consistency
Semantic interpretations strictly state *"Change overlapping [Class]"*. Cross-verification confirms $100\%$ logical consistency between $T_1/T_2$ class shifts and assigned change types.

---

## 20. Limitations
1. OSCD provides binary change annotations only.
2. 10m GSD resolution limits detection of objects $< 10\text{m} \times 10\text{m}$.
3. Seasonal crop shifts and solar angle variations induce false positives.
4. System cannot confirm real-world construction or demolition events without ground truth.

---

## 21. Future Work
1. Integrate multi-temporal Siamese CNN or Transformer architectures given GPU hardware.
2. Incorporate dense semantic segmentation datasets (e.g., SpaceNet, LandCoverAI).
3. Implement cloud and atmospheric correction pipelines (e.g., Sen2Cor).

---

## 22. Conclusion
The project successfully developed and validated a CPU-friendly semantic satellite change detection system. By combining 32 tabular features, a tuned XGBoost classifier, and a rule-based spectral semantic layer, the system provides explainable, object-level change insights while adhering to strict scientific qualification standards.

---

## 23. References
1. Daudt, R. C., et al. "Urban Change Detection for Multispectral Earth Observation Images Using Convolutional Neural Networks." IGARSS 2018. (OSCD Dataset).
2. Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System." KDD 2016.
3. Rouse, J. W., et al. "Monitoring vegetation systems in the Great Plains with ERTS." NASA SP-351, 1974. (NDVI).
4. Zha, Y., et al. "Use of normalized difference built-up index (NDBI) in calculating urban areas." International Journal of Remote Sensing, 2003.
