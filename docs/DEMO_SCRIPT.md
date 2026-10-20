# Live Demonstration Script (3-5 Minutes)

**Title:** AI/ML-Based Semantic Satellite Change Detection  
**Target Audience:** Technical & Evaluator Panel  

---

## 1. Problem Statement & Motivation (30 seconds)
> *"Satellite imagery provides critical monitoring for urban expansion, disaster relief, and infrastructure development. However, raw binary change maps only tell us WHERE change happened, not WHAT physical land-cover class changed. Our system detects surface changes using XGBoost and categorizes them into physical classes like Roads, Buildings, Water, and Vegetation."*

---

## 2. System Architecture & Dataset (45 seconds)
> *"We use 10m Sentinel-2 multi-temporal imagery from the ONERA Satellite Change Detection (OSCD) dataset. From bi-temporal image pairs (T1 and T2), our feature extraction engine computes 32 spectral, spatial, and temporal features including NDVI, NDWI, and NDBI. An XGBoost classifier predicts pixel-level change probabilities at an optimal threshold of 0.72."*

---

## 3. Live Web Application Demo (90 seconds)
> *"Here is our Flask web application running on CPU. We select benchmark scene `test_000` and click 'Analyze Selected Sample Scene'."*
>
> 1. *"Notice the 5 summary cards: status, changed pixel count, changed region count, largest region, and total runtime under 0.4 seconds."*
> 2. *"Next, examine our 6-panel visual dashboard: Before (T1 RGB), After (T2 RGB), XGBoost Change Mask, Land Cover Mask, Semantic Change Map, and the transparent overlay."*
> 3. *"Look at the Class-Specific Change Breakdown: road changes account for 49.9%, bare land for 39.6%, and buildings for 3.7%."*
> 4. *"The Connected Component Analysis table lists all 145 detected changed regions with exact pixel areas, bounding box coordinates, and centroids."*

---

## 4. Scientific Rigor & Limitations (45 seconds)
> *"We maintain strict scientific credibility: OSCD provides binary change ground truth. Our semantic classes are derived via spectral index rules and distance transform thinness heuristics. Therefore, we explicitly report results as 'change overlapping road class' rather than unverified construction claims."*

---

## 5. Conclusion & Deliverables (30 seconds)
> *"Users can download complete results as JSON, CSV, or ZIP packages. The entire project is CPU-compatible, deterministic, and fully automated via REST API."*
