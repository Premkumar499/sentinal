# OSCD Dataset Documentation

## 1. Dataset Overview
- **Name:** ONERA Satellite Change Detection (OSCD)
- **Source:** ONERA / ESA Sentinel-2 Imagery
- **Resolution:** 10m Ground Sample Distance (GSD)
- **Problem Formulation:** Bi-temporal satellite change detection between Date 1 ($T_1$) and Date 2 ($T_2$).

---

## 2. Band Mapping & Spectral Representation

| Sentinel-2 Band | Spectral Name | Resolution | Pipeline Band Key |
| :--- | :--- | :--- | :--- |
| **B02** | Blue (490 nm) | 10m | `blue` |
| **B03** | Green (560 nm) | 10m | `green` |
| **B04** | Red (665 nm) | 10m | `red` |
| **B08** | Near Infrared (842 nm) | 10m | `nir` |
| **B11** | Short-wave Infrared (1610 nm) | 20m (Resampled to 10m) | `swir` |

---

## 3. Label Mask Characteristics
- **Binary Labels:** `0 = Unchanged`, `1 = Changed`
- **Annotation Scope:** Human-annotated urban and environmental changes (e.g., new earthworks, construction, vegetation clearance).
- **Semantic Scope Note:** The OSCD dataset contains **binary change labels only**. Physical object semantic classes (Roads, Buildings, Water, Vegetation) are inferred via spectral index rules and morphological distance transforms in the Day 4/5 pipeline.
