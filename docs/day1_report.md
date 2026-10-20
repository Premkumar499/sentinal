# Day 1 Technical Report: Dataset Exploration & Baseline Difference Pipeline

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Execution Environment:** CPU Only  

---

## 1. Executive Summary

Day 1 established the foundational data ingestion and baseline visualization pipeline for multi-temporal satellite images from the OSCD dataset. We analyzed bi-temporal Sentinel-2 image pairs ($T_1$ and $T_2$), inspected ground-truth change masks, mapped multispectral bands, verified spatial resolutions (10m GSD), and built a simple difference map baseline.

---

## 2. Key Accomplishments

1. **OSCD Dataset Structure & Mapping**:
   - Identified 13 multispectral bands per scene.
   - Standardized 10m GSD band mapping: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR).
2. **Reflectance Preprocessing**:
   - Implemented `OSCDLoader` and `PreprocessingPipeline` in `src/preprocessing/pipeline.py`.
   - Scaled raw digital numbers (DN / 10,000) to normalized reflectance $[0.0, 1.0]$.
   - Handled NaNs and missing values with zero-fill.
3. **Baseline Difference Map**:
   - Computed simple pixel-wise Euclidean difference across RGB+NIR bands between $T_1$ and $T_2$.
   - Generated initial visual change maps to establish the problem baseline.
