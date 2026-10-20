# Technical & Conceptual System Explanation

---

## 1. Simple Non-Technical Explanation (For General Audience)

Imagine taking two photos of a city from space—one taken in 2021 and another in 2024. 

Our computer system looks at both photos side-by-side:
1. **Detecting Changes:** It highlights areas where colors, textures, or shapes have significantly changed (e.g., green forest turning into brown dirt or grey concrete).
2. **AI Probability:** An AI algorithm (**XGBoost**) scores every single dot (pixel) from 0% to 100% based on how confident it is that a real surface change occurred.
3. **Land Cover Category:** The system analyzes light reflections (infrared and visible light) to figure out if the changed area matches a **Road**, **Building**, **Water**, **Vegetation**, or **Bare Land**.
4. **Summary & Inspection:** It draws red/yellow/blue boxes around major changes, ranks them by size and significance, and displays them on an easy-to-use web dashboard.

---

## 2. Technical System Explanation (For Engineers & Reviewers)

```
[ INPUT ]  ──►  Pair of Sentinel-2 Multispectral Images (T1 & T2)
                   │
[ FEATURES ] ──►  32 Tabular Features per Pixel (Spectral Diffs, Ratios, Sobel, Texture)
                   │
[ MODEL ]   ──►  XGBoost Classifier (P(Change) >= 0.72)
                   │
[ BINARY ]  ──►  Morphological 3x3 Close + CCA Filtering (Area >= 3 px)
                   │
[ SEMANTIC ] ──►  Spectral Index Rules (NDVI/NDWI/NDBI) & Thinness Heuristics
                   │
[ REGIONS ] ──►  Bounding Boxes, Significance Scores & Change-Type Categorization
                   │
[ OUTPUT ]  ──►  Flask Web UI Dashboard + REST API (POST /api/detect)
```

### Detailed Pipeline Components:

1. **Preprocessing & Band Loading:**
   Loads Blue (B02), Green (B03), Red (B04), NIR (B08), and SWIR (B11) rasters. Normalizes top-of-atmosphere reflectance values to $[0.0, 1.0]$.
2. **Feature Engineering Engine:**
   Computes 32 channel features per pixel:
   - Absolute spectral differences: $\Delta C = |C_{T2} - C_{T1}|$
   - Band reflectance ratios: $C_{T2} / (C_{T1} + \epsilon)$
   - Normalized indices: $\Delta\text{NDVI}$, $\Delta\text{NDBI}$
   - Local texture statistics: $3 \times 3$ sliding window mean and standard deviation
   - Spatial edge magnitudes: Sobel filter gradient magnitude
   - Built-up distance transform
3. **XGBoost Inference:**
   Predicts continuous change probabilities $P(\text{Change}) \in [0.0, 1.0]$. Applied binary mask at threshold $0.72$.
4. **Semantic Layer Intersection:**
   Intersecting binary change mask with spectral land-cover rules:
   - Water: $\text{NDWI} > 0.1$
   - Vegetation: $\text{NDVI} > 0.3$
   - Built-up (Road/Building): $\text{NDBI} > 0.0$ (Thinness $\le 5\text{px} \rightarrow \text{Road}$; $> 5\text{px} \rightarrow \text{Building}$)
   - Bare Land: Low NDVI, low NDBI, low NDWI
5. **Region Extraction & Ranking:**
   OpenCV Connected Component Analysis extracts bounded regions $(X, Y, W, H)$, calculates Significance Scores, and categorizes qualitative change types (`New`, `Removed`, `Expanded`, `Transition`, `Unknown`).
