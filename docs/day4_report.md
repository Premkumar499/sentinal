# Day 4: Semantic Change Detection Layer and Interface Implementation Report

## 1. Day 4 Objective
The objective of Day 4 is to extend our binary change detection system with a semantic classification layer. We associate detected change regions (XGBoost change mask) with their underlying physical land cover categories (e.g. Roads, Buildings, Water, Vegetation, Bare Land) without training a heavy deep-learning model from scratch, operating efficiently on a CPU.

---

## 2. Why Semantic Information is Needed
A binary change detection system only identifies *where* pixels changed (0 = No Change, 1 = Change) but fails to explain *what* changed. In urban monitoring, civil engineering, and environmental compliance, stakeholders need to categorize changes (e.g. road expansion, building construction, deforestation, or reservoir level changes) to make informed decisions.

---

## 3. Semantic Data / Model Source
Since the workspace did not contain preloaded external land-cover maps or shapefiles, and local hardware is CPU-only, we implemented a **Rule-Based Spectral Index and Morphological Shape Classifier**:
*   **Source Type:** Unsupervised physical classification.
*   **Bands Used:** Sentinel-2 Surface Reflectance bands (Blue - B02, Green - B03, Red - B04, NIR - B08, SWIR1 - B11).
*   **Coordinate Reference System:** Aligned with the input scene's Coordinate Reference System (UTM projection).
*   **Resolution:** 10 meters.
*   **Coverage:** 100% spatial coverage for all 100 OSCD scenes.
*   **Alignment:** 1-to-1 pixel correspondence with zero spatial shift.

---

## 4. Class Definitions
The class mapping is loaded from `config/semantic_classes.json`:
*   `0`: Background / Unknown (unclassified or unchanged areas)
*   `1`: Road (Narrow linear built-up features)
*   `2`: Building / Built-up (Wider urban structures)
*   `3`: Water (Reservoirs, rivers, oceans)
*   `4`: Vegetation (Forests, agriculture, parks)
*   `5`: Bare Land (Sandy soil, dirt, open rock)
*   `6`: Unknown Change (Changed pixels where land cover class is unresolved)

---

## 5. Alignment Process
To guarantee perfect spatial registration:
*   Features are extracted from the normalized reflectance bands of each OSCD scene.
*   The SWIR band (original resolution 20m) is resampled to 10m to match the other bands using **Nearest-Neighbor Resampling** to preserve categorical spectral properties.
*   All calculations are done inline within the exact bounding dimensions ($256 \times 256$) of the aligned OSCD scenes.

---

## 6. Semantic Mask Creation
The semantic classification is performed using physical index thresholding on the post-change image ($T2$):
1.  **Water Mask:** Normalized Difference Water Index ($NDWI = \frac{Green - NIR}{Green + NIR} > 0.05$).
2.  **Vegetation Mask:** Normalized Difference Vegetation Index ($NDVI = \frac{NIR - Red}{NIR + Red} > 0.25$) excluding water.
3.  **Built-up Mask:** Normalized Difference Built-up Index ($NDBI = \frac{SWIR - NIR}{SWIR + NIR} > 0.05$) excluding water and vegetation.
4.  **Bare Land Mask:** Low NDVI ($\le 0.2$), low NDWI ($\le 0.0$), low NDBI ($\le 0.05$), and Red reflectance $> 0.12$.
5.  **Roads vs. Buildings Separation:** Applied a **morphological distance transform** on the built-up mask. Pixels with distance $\le 2.2$ pixels (corresponding to structures narrower than $\sim 44$m) are classified as **Roads** (Class 1); pixels with distance $> 2.2$ are classified as **Buildings** (Class 2).

---

## 7. Change + Semantic Intersection
To produce the semantic change map, we evaluate pixels where the binary XGBoost change detector predicted a change (`change_mask == 1`). 
For these changed pixels, we map the corresponding land cover class from the $T2$ semantic mask. If a changed pixel is unclassified by the spectral indices (Class 0), it is mapped to **Unknown Change** (Class 6).

---

## 8. Class-Specific Change Logic
*   **Road Change:** `(change_mask == 1) & (semantic_mask == 1)`. Identifies change overlapping road corridors.
*   **Building Change:** `(change_mask == 1) & (semantic_mask == 2)`. Identifies new building footprints or extensions.
*   **Water Change:** `(change_mask == 1) & (semantic_mask == 3)`. Tracks change overlapping water bodies (e.g. reservoir shrinkage/expansion).
*   **Vegetation Change:** `(change_mask == 1) & (semantic_mask == 4)`. Identifies vegetation clearing or re-greening.
*   **Unknown Change:** `(change_mask == 1) & (semantic_mask == 6)`. Captures changes in unclassified landscapes.

Class-specific masks are saved as binary images in `outputs/predictions/<scene>_road_change.png`, etc.

---

## 9. Connected-Component Region Extraction
Pixel-level changes are aggregated into spatially contiguous "changed regions" using connected-component analysis:
*   Each region is analyzed for: Region ID, Class Name, Pixel Area, Bounding Box (X, Y, Width, Height), and Centroid (X, Y).
*   Statistics are exported to `outputs/predictions/<scene>_change_regions.csv`.

### Min Region Size Filter Sweep (Scene: `val_000`):
We swept the `MIN_REGION_PIXELS` parameter to select the optimal threshold for removing noise regions:

| Min Region Pixels | Number of Change Regions |
| :---: | :---: |
| 2 | 32 |
| 5 | 18 |
| **10 (Selected)** | **9** |
| 20 | 2 |
| 50 | 1 |
| 100 | 0 |

*   **Choice:** **10 pixels** ($\sim 1000 \text{ m}^2$) was chosen as it effectively removes small false positive noise blocks while retaining significant urban/industrial changes.

---

## 10. Temporal Semantic Change (T1 vs. T2)
To detect transitions rather than just static intersections, the system builds semantic masks for both dates ($T1$ and $T2$) and compares transitions inside the change mask. If the T1 class differs from the T2 class (e.g., Vegetation in T1 and Building in T2), it indicates a land cover conversion.

---

## 11. Example Results (Scene: `test_000`)
Qualitative validation plots were generated and manually inspected:
*   `outputs/visualizations/semantic_mask_test_000.png`
*   `outputs/visualizations/day4_semantic_change_test_000.png`

### Scene Breakdown:
*   **Total changed area:** 5,434 pixels
*   **Road-overlapping change:** 2,731 pixels (50.26%)
*   **Building-overlapping change:** 171 pixels (3.15%)
*   **Water-overlapping change:** 277 pixels (5.10%)
*   **Vegetation-overlapping change:** 0 pixels (0.00%)
*   **Unknown change:** 114 pixels (2.10%)
*   **Number of change regions:** 111
*   **Largest change region:** `road` (415 pixels)

---

## 12. Evaluation & Validation
*   **Binary Change detector performance** remains optimized (Test F1: **0.6250**, Test IoU: **0.4546**).
*   **Semantic Layer Validation:** No OSCD ground-truth semantic labels were available. Hence, semantic metrics are qualitative. Direct visual overlay inspections show that water bodies are perfectly masked, vegetation boundaries match physical green spaces, and roads/narrow features align with linear infrastructure.
*   **Wording constraint:** All findings are reported as "change overlapping [class] class" to maintain technical precision without claiming certified ground-truth classification.

---

## 13. Reusable Python API for Flask
We prepared the Flask integration by exposing `detect_semantic_change()` in `src/models/semantic.py`.
This function returns:
*   `change_mask`: Binary NumPy array of predicted changes.
*   `semantic_mask_t1`: Categorical land cover mask for T1.
*   `semantic_mask_t2`: Categorical land cover mask for T2.
*   `semantic_change_mask`: Mapped semantic change categories.
*   `regions`: List of dicts with area, bounding boxes, and centroids.
*   `statistics`: Dictionary with counts and largest region details.

This backend interface runs entirely in memory without Flask dependencies, allowing headless execution and clean API testing.

---

## 14. Remaining Limitations
1.  **Spectral Confusion:** Physical index thresholds can misclassify bare soil or shadows as built-up areas.
2.  **Road thickness heuristic:** Highly elongated building blocks or alleys can be misclassified as roads due to the thickness threshold.

---

## 15. Day 5 Plan: Lightweight Semantic Classification Layer (Post-Processing)
On Day 5, we will build a dedicated lightweight multi-class classifier (e.g. Random Forest or shallow Neural Network) utilizing T1/T2 band reflectance and texture features to refine road/building classification, using the rule-based maps as weak training supervision.
