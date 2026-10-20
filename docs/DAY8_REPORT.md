# Day 8 Technical Report: Semantic Region Interpretation, Change-Type Categorization & Explainability

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Primary ML Model:** XGBoost Tuned Classifier (Threshold = 0.72)  
**Semantic Layer:** Spectral Index Rules (NDVI/NDWI/NDBI) + Distance Transform Heuristics  
**Execution Environment:** 100% CPU Compatible  

---

## 1. Executive Summary

Day 8 extended the system from pixel-level binary change maps to object-level, structured semantic region interpretation. Rather than simply displaying *"CHANGE"*, the system now answers critical questions: *Where did the change occur? How large is it? Which land cover class overlaps the change? What is the qualitative change type (New, Removed, Expanded, Transition)? How significant is the region?*

All interpretations adhere strictly to scientific qualification rules (e.g., *"Change overlapping road class"* rather than unverified real-world construction assertions).

---

## 2. Day 8 System Audit Summary

| Component | Status | Verification Result |
| :--- | :--- | :--- |
| **Change Detection** | **PASS** | XGBoost model predicts change probabilities at threshold 0.72 |
| **Semantic Classification** | **PASS** | Spectral index rules generate 5 land cover classes + Unknown class |
| **Region Extraction** | **PASS** | Connected Component Analysis extracts bounded regions & shape metrics |
| **Flask UI & API** | **PASS** | Dashboard updated with change type filters, detail views & REST API |

---

## 3. Qualitative Change-Type Categorization

By comparing $T_1$ and $T_2$ land-cover semantic masks across changed pixels, the system categorizes region-level change types:

| Change Type | Definition | Example Scenario |
| :--- | :--- | :--- |
| **New** | Class absent in $T_1$ ($c_1 = 0$), present in $T_2$ ($c_2 \neq 0$) | Background/Bare Land $\rightarrow$ Road/Building |
| **Removed** | Class present in $T_1$ ($c_1 \neq 0$), absent in $T_2$ ($c_2 = 0$) | Building $\rightarrow$ Demolished / Cleared Ground |
| **Expanded** | Class present in both $T_1$ and $T_2$ ($c_1 = c_2 \neq 0$) | Pre-existing Road $\rightarrow$ Widened Road region |
| **Transition** | Class shifts between distinct non-zero categories | Vegetation $\rightarrow$ Built-up Structure |
| **Unknown** | Change detected, but semantic mask is unclassified ($c = 6$) | Low spectral confidence or ambiguous pixels |

---

## 4. Class-Specific Change Analysis

1. **Road Change Analysis:** Identifies linear built-up features (thickness $\le 5$ pixels). Outputs *"Change overlapping road class (New/Expanded)"*.
2. **Building Change Analysis:** Identifies compact built-up structures (thickness $> 5$ pixels). Outputs *"Building-overlapping change detected"*.
3. **Water Change Analysis:** Identifies NDWI shifts ($> 0.05$). Tracks water expansion/reduction without asserting cause.
4. **Vegetation Change Analysis:** Evaluates NDVI differences ($\Delta\text{NDVI}$). Differentiates vegetation gain vs loss.

---

## 5. Region Significance Score & Shape Analysis

- **Significance Score Formula:**
  $$\text{Significance Score} = \left( \frac{\text{Area (pixels)}}{100} \right) \times P_{\text{mean}}$$
  where $P_{\text{mean}}$ is the mean XGBoost change probability across the region.
- **Geometric Properties:** Area, Bounding Box $(X, Y, W, H)$, Centroid $(CX, CY)$, Aspect Ratio ($W/H$), and Compactness ($\text{Area} / (W \times H)$).
- **Quality Flags:**
  ```json
  {
    "alignment_ok": true,
    "semantic_available": true,
    "small_region": false,
    "low_change_confidence": false
  }
  ```

---

## 6. Generated Reports & Data Outputs

1. **Transition Matrix CSV (`outputs/reports/semantic_transition_matrix.csv`)**:
   Tracks pixel counts for all $T_1 \rightarrow T_2$ class transitions.
2. **Semantic Change Regions CSV (`outputs/reports/semantic_change_regions.csv`)**:
   Includes Scene ID, Region ID, Rank, Class, $T_1$ Class, $T_2$ Class, Change Type, Area, Significance Score, Mean Probability, $\Delta\text{NDVI}$, $\Delta\text{NDBI}$, Bounding Box, Centroid, and Quality Flags.
3. **Region Summary JSON (`region_summary.json`)**:
   Structured JSON exported for each run.
4. **Region Boundary Overlays & 4-Crop Panels**:
   - `<run_id>_region_overlay.png`: T2 image overlaid with color-coded bounding boxes and region labels.
   - `crops/region_<id>_crop.png`: Per-region 4-panel comparison ($T_1$ crop, $T_2$ crop, Change crop, Semantic crop).

---

## 7. Flask UI & API Enhancements

- **Interactive Results Page (`app/templates/results.html`)**:
  - Class filter dropdown (*All, Road, Building, Water, Vegetation, Bare Land, Unknown*).
  - Change Type filter dropdown (*All, New, Removed, Expanded, Transition, Unknown*).
  - Regional rank badges and Significance Scores.
- **New Region Details Inspection Route (`/region/<run_id>/<region_id>`)**:
  - Displays individual 4-crop comparison panels, geometric shape metrics, spectral index deltas, and quality flags.
- **REST API Update (`POST /api/detect`)**:
  - Returns structured region list with change types, bounding boxes, centroids, scores, and transition matrix top pairs.

---

## 8. Unit & End-to-End Automated Testing

- **New Test Suite (`tests/test_day8_region_analysis.py`)**:
  - Verified `determine_change_type` mapping.
  - Verified synthetic transition matrix pixel counts.
  - Verified region ranking, significance scores, and quality flags.
- **Test Pass Rate:** **42 / 42 Tests Passed** cleanly in `pytest`.

---

## 9. Final Technical Conclusion

"Day 8 extended the semantic change-detection system from pixel-level change visualization to region-level change interpretation. Detected changes were associated with available semantic classes, analyzed temporally where T1/T2 semantic information was available, grouped into significant regions, and summarized using area, location, change probability, semantic class, and change type. The system explicitly preserves unknown and uncertain regions and avoids unsupported claims about real-world events."
