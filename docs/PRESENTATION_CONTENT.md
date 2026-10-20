# Presentation Content (15 Slides Outline)

---

### Slide 1: Title Slide
- **Title:** Semantic Satellite Change Detection System
- **Subtitle:** AI/ML Multi-Temporal Change Detection & Land-Cover Region Interpretation
- **Target Stack:** OSCD Dataset • XGBoost ML Core • 100% CPU Compatible • Flask Framework

---

### Slide 2: Problem Statement
- **Problem:** Earth observation generates massive multi-temporal satellite imagery, but manual change inspection is slow and resource-intensive.
- **Challenge:** Traditional difference maps tell you *where* something changed, but fail to explain *what class* overlaps the change. Deep learning models often require heavy GPUs.

---

### Slide 3: Motivation & Real-World Use Cases
- Infrastructure & urban expansion monitoring
- Agricultural & land-cover transformation tracking
- Disaster response & surface anomaly observation
- *Constraint:* System must run efficiently on standard CPU hardware without GPU reliance.

---

### Slide 4: Core Project Objectives
1. Implement a 32-feature extraction pipeline for Sentinel-2 satellite images.
2. Train a CPU-friendly **XGBoost classifier** to predict change probabilities.
3. Intersect changes with a **Spectral Rule Semantic Layer** (NDVI, NDWI, NDBI).
4. Extract bounded regions, rank by **Significance Score**, and categorize change types (`New`, `Expanded`, `Removed`, `Transition`).
5. Deliver an interactive **Flask Web UI** and **REST API**.

---

### Slide 5: Dataset (OSCD)
- **Dataset:** ONERA Satellite Change Detection (OSCD)
- **Extents:** 100 aligned Sentinel-2 image pairs (10m GSD) across global urban/natural scenes.
- **Ground Truth:** Binary change masks (0 = Unchanged, 1 = Changed).
- **Split:** 60 Train / 20 Validation / 20 Test scenes.

---

### Slide 6: System Architecture
- $T_1 + T_2 \rightarrow$ 32 Feature Extraction $\rightarrow$ XGBoost Inference ($P \ge 0.72$) $\rightarrow$ Morphological Close + CCA $\rightarrow$ Spectral Rules Intersect $\rightarrow$ Region Analysis $\rightarrow$ Flask Web UI / REST API.

---

### Slide 7: Feature Engineering (32 Features)
- Spectral Band Differences ($\Delta\text{Red}, \Delta\text{Green}, \Delta\text{Blue}, \Delta\text{NIR}, \Delta\text{SWIR}$)
- Normalized Index Shifts ($\Delta\text{NDVI}, \Delta\text{NDBI}$)
- Band Ratios & Local Texture ($3 \times 3$ mean and std)
- Spatial Edges (Sobel magnitude) & Built-up Distance Transforms

---

### Slide 8: XGBoost Model Selection & Tuning
- **Why XGBoost?** Fast CPU inference ($< 0.40$s/scene), interpretable feature importance, handles tabular pixel vectors efficiently.
- **Threshold Optimization:** Selected $0.72$ based on validation set F1 optimization.
- **Class Balancing:** `scale_pos_weight` set to negative/positive pixel ratio.

---

### Slide 9: Semantic Layer & Qualitative Change Types
- **Classes:** Road, Building, Water, Vegetation, Bare Land, Unknown.
- **Rules:** Spectral indices (NDVI/NDWI/NDBI) + morphological thinness ($\le 5\text{px} \rightarrow \text{Road}$; $> 5\text{px} \rightarrow \text{Building}$).
- **Change Types:** `New`, `Removed`, `Expanded`, `Transition`, `Unknown`.

---

### Slide 10: Quantitative Validation Results (Day 9)
- **F1-Score:** **$0.4606 \pm 0.2125$** (Median: **$0.4591$**)
- **IoU Score:** **$0.3244 \pm 0.1835$** (Median: **$0.3004$**)
- **Precision:** **$0.5621$** | **Recall:** **$0.4635$**
- **Improvement over Simple Difference Baseline:** **$+110.9\%$ F1 Improvement** ($0.4606$ vs $0.2184$).

---

### Slide 11: Visual Results & 6-Panel Grids
- Showcase $T_1$ RGB, $T_2$ RGB, XGBoost Change Map, Color-Coded Semantic Map, Semantic Change Map, and Bounding Box Overlays.

---

### Slide 12: Error Analysis & Category Distribution
- **False Positives:** Seasonal vegetation shifts ($38.2\%$), Illumination angles ($25.5\%$), Shadows ($16.4\%$).
- **False Negatives:** Small objects $< 10\text{m}$ ($41.2\%$), Low contrast built-up ($25.9\%$).

---

### Slide 13: Web Application & REST API
- Modern dark glassmorphism UI dashboard.
- Interactive table filters by Class and Change Type.
- Region Detail Inspection Page (`/region/<run_id>/<region_id>`) with 4-panel crops.
- Programmatic REST API (`POST /api/detect`).

---

### Slide 14: System Limitations & Scientific Honesty
- Cannot confirm physical construction or demolition without ground truth.
- Cannot confirm flood events from spectral shifts alone.
- Resolution boundary: 10m GSD cannot resolve objects $< 10\text{m} \times 10\text{m}$.
- Wording Rule: Outputs state *"Change overlapping [Class]"*.

---

### Slide 15: Conclusion & Future Work
- **Conclusion:** Successful development and scientific validation of a CPU-friendly, explainable satellite change detection prototype.
- **Future Work:** Deep learning Siamese CNN/Transformers (given GPU hardware), dense semantic datasets, cloud/atmospheric correction pipelines.
