# System Capabilities: What the Model Can Do

**Project Name:** Semantic Satellite Change Detection  
**Evaluation Status:** Scientifically Validated (Day 9)  

---

## 1. Primary Operational Capabilities

The **Semantic Satellite Change Detection System** is an end-to-end, CPU-compatible machine learning platform capable of:

1. **Multi-Temporal Satellite Scene Comparison:** Ingesting paired Sentinel-2 multispectral satellite images ($T_1$ and $T_2$) across aligned spatial extents.
2. **Deterministic Feature Extraction:** Extracting 32 spectral difference, ratio, texture (3x3 local mean/std), edge (Sobel magnitude), and distance transform features.
3. **Probability Map Generation:** Predicting continuous pixel-level change probabilities $P(\text{Change}) \in [0.0, 1.0]$ using a tuned XGBoost classifier.
4. **Binary Change Masking:** Thresholding change probabilities at $0.72$ combined with morphological filtering ($3 \times 3$ closing + connected component filtering $\ge 3$ pixels).
5. **Semantic Layer Intersection:** Associating detected change pixels with physical land-cover categories (Roads, Buildings, Water, Vegetation, Bare Land, Unknown) using spectral rules (NDVI/NDWI/NDBI) and morphological distance transforms.
6. **Object-Level Region Analysis:** Grouping changed pixels into connected component regions, generating bounding boxes $(X, Y, W, H)$, centroids $(CX, CY)$, aspect ratios, compactness, and Significance Scores.
7. **Temporal Change-Type Categorization:** Categorizing region-level shifts into qualitative change types (`New`, `Removed`, `Expanded`, `Transition`, `Unknown`).
8. **Interactive Dashboard & REST API:** Serving predictions, 6-panel visual grids, region tables, dynamic filters, region detail inspection pages (`/region/<run_id>/<region_id>`), and automated REST API endpoints (`POST /api/detect`).

---

## 2. Quantitative Benchmark Performance

- **Evaluated Test Set:** 20 unseen OSCD scenes (`test_000` through `test_019`).
- **Mean F1-Score:** $0.4606 \pm 0.2125$ (Median: $0.4591$).
- **Mean IoU:** $0.3244 \pm 0.1835$ (Median: $0.3004$).
- **Mean Precision:** $0.5621 \pm 0.1699$.
- **Mean Recall:** $0.4635 \pm 0.2833$.
- **CPU Inference Latency:** $< 0.40$ seconds per $600 \times 600$ scene.
