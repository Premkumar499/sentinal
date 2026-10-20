# Day 12 Technical Report: Geospatial Localization, Region Inspection & Interactive Mapping

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Primary Model:** XGBoost Tuned Classifier (Threshold = 0.72, Frozen)  
**Geospatial Layer:** `src/geospatial.py` (Rasterio metadata, real-world area, GeoJSON export, boundary overlays)  

---

## 1. Executive Summary & Objective

Day 12 extended the Semantic Satellite Change Detection system from:
$$\text{DETECT} \longrightarrow \text{UNDERSTAND} \longrightarrow \text{SEARCH}$$
to:
$$\text{DETECT} \longrightarrow \text{UNDERSTAND} \longrightarrow \text{SEARCH} \longrightarrow \text{LOCATE} \longrightarrow \text{INSPECT} \longrightarrow \text{REPORT}$$

Analysts can search for specific change types, locate them on an interactive spatial boundary map, inspect detailed 4-panel crops ($T_1$, $T_2$, Change, Semantic), understand why each region was selected, and export structured GeoJSON/CSV results.

Zero model retraining or hyperparameter modifications were performed during Day 12, preserving scientific integrity.

---

## 2. Geospatial Metadata & Coordinate Transformations (`src/geospatial.py`)

- **Geospatial Metadata Inspection:** Reads CRS, affine transform matrix, bounds, and pixel resolution via Rasterio.
- **Coordinate Conversion:** `pixel_to_geo(x, y)` converts pixel coordinates to (latitude, longitude) when valid CRS exists. For unreferenced images, safely falls back to `coordinate_type = "pixel"` without fabricating GPS coordinates.
- **Real-World Area Calculation:** Converts pixel count to $m^2$ and hectares based on spatial resolution ($10\text{m} \times 10\text{m} \rightarrow 100\text{m}^2/\text{pixel}$).
- **GeoJSON FeatureCollection:** Generates standard GeoJSON polygons (`outputs/geospatial/<scene>_regions.geojson`) for georeferenced scenes.
- **Boundary Overlays:** Draws color-coded class contours and Region IDs over $T_2$ RGB image (`outputs/geospatial/<scene>_region_boundaries.png`).

---

## 3. Interactive Flask Map View (`/map`, `app/templates/map.html`)

- **Interactive Canvas Map:** Renders image-space boundary overlays with real-time class, change type, and minimum area filtering.
- **Region Inspection Sidebar:** Clicking any bounding box dynamically updates the sidebar showing Region ID, Class, Change Type, Area ($px$ and $m^2$), Change Score, Centroid, Bounding Box, and a **"View Full 4-Panel Region Crops & Details"** button.

---

## 4. Enhanced Region Detail Page (`/region/<run_id>/<region_id>`)

- **"Why Was This Region Selected?" Card:** Displays step-by-step query matching rules:
  1. Change probability exceeding threshold ($72.0\%$).
  2. Spatial area exceeding minimum filter ($10\text{ px}$).
  3. Overlap with target semantic class.
  4. Temporal land cover transition categorization.
- **Qualified Interpretation Story:** Expressed using cautious, scientifically sound language (e.g. *"Detected change overlaps a road-class region"*).

---

## 5. Verification & Testing

- **Geospatial Unit Test Suite (`tests/test_day12_geospatial.py`):** Verified metadata inspection, area conversion, boundary rendering, GeoJSON fallback, and Flask map/API endpoints.
- **System Test Suite:** **50 / 50 pytest cases PASSED** cleanly.

---

## 6. Final Technical Conclusion

"Day 12 extended the semantic satellite change-detection prototype with region-level geospatial and inspection capabilities. Detected semantic change regions can now be searched, filtered, located in image or geographic space when valid metadata exists, inspected through aligned T1/T2 crops and change maps, and exported as structured results. The system therefore progresses from pixel-level change detection toward searchable and interpretable change information while preserving uncertainty and data-source limitations."
