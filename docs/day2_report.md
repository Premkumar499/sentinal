# Day 2 Technical Report: XGBoost Binary Change Detection Baseline

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Model:** XGBoost Baseline Classifier  
**Execution Environment:** CPU Only  

---

## 1. Executive Summary

Day 2 constructed the initial machine learning pipeline for binary satellite change detection using XGBoost. We engineered multi-temporal spectral and ratio features from $T_1$ and $T_2$ images, extracted tabular pixel datasets, addressed class imbalance (ground truth changes represent <15% of pixels), trained an XGBoost binary classifier, and evaluated binary change detection performance.

---

## 2. Key Accomplishments

1. **Feature Engineering**:
   - Extracted spectral band values ($T_1$ and $T_2$), absolute band differences ($|T_2 - T_1|$), band ratios, spectral indices (NDVI, NDWI, NDBI), and index differences.
2. **Model Training & Class Imbalance**:
   - Implemented `scale_pos_weight` in XGBoost to handle class imbalance between changed and unchanged pixels.
   - Saved baseline model weights to `models/xgboost_change_detector.json`.
3. **Evaluation**:
   - Evaluated binary change detection maps against OSCD ground truth masks.
