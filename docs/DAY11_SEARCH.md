# Day 11: Semantic Change Searching & Natural Language Interpretation Engine

**Project Name:** Semantic Satellite Change Detection  
**Module Name:** Day 11 Semantic Region Search & Query Engine  
**Database:** SQLite (`data/change_regions.db`)  
**Status:** Validated & Fully Integrated (Zero ML retraining)  

---

## 1. Why Semantic Search?
While binary change detection identifies *where* spectral differences occurred, analysts frequently require targeted queries such as *"Show road changes"* or *"Find large building changes"*. Day 11 extends the pipeline from passive visualization to **active semantic change querying** without modifying the underlying frozen XGBoost model.

---

## 2. Region Data Model & SQLite Database Schema (`data/change_regions.db`)

Each detected region is indexed in SQLite table `change_regions`:

```sql
CREATE TABLE change_regions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    region_id INTEGER NOT NULL,
    semantic_class TEXT NOT NULL,
    change_type TEXT NOT NULL,
    area_pixels INTEGER NOT NULL,
    change_score REAL NOT NULL,
    centroid_x INTEGER NOT NULL,
    centroid_y INTEGER NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    t1_class TEXT NOT NULL,
    t2_class TEXT NOT NULL
);
CREATE INDEX idx_semantic_class ON change_regions(semantic_class);
CREATE INDEX idx_change_type ON change_regions(change_type);
CREATE INDEX idx_scene_id ON change_regions(scene_id);
```

---

## 3. Natural Language Query Parser (`QueryParser`)

The rule-based query parser maps natural language inputs to structured query parameters:

| Example User Query | Extracted Intent | SQL Filter Applied |
| :--- | :--- | :--- |
| `"Show road changes"` | `class = "road"` | `WHERE semantic_class = 'road'` |
| `"Find building changes"` | `class = "building"` | `WHERE semantic_class = 'building'` |
| `"Show water changes"` | `class = "water"` | `WHERE semantic_class = 'water'` |
| `"Show large changes"` | `min_area = 200, sort = "area_desc"` | `WHERE area_pixels >= 200 ORDER BY area_pixels DESC` |
| `"Find new road regions"` | `class = "road", change_type = "new"` | `WHERE semantic_class = 'road' AND change_type = 'new'` |
| `"Find airport changes"` | `status = "unsupported"` | Returns helpful message listing supported classes |

---

## 4. REST API Reference

### A. Natural Language Search Endpoint
- **URL:** `POST /api/search`
- **Body:** `{"query": "show road changes"}`
- **Response:**
```json
{
  "status": "success",
  "query": "show road changes",
  "intent": {
    "semantic_class": "road",
    "change_type": "all",
    "min_area": null,
    "sort_by": "area_desc"
  },
  "summary": {
    "matching_regions": 12,
    "total_matching_area": 14200,
    "largest_region_area": 2400,
    "most_common_class": "road"
  },
  "results": [...]
}
```

### B. Structured Filter Endpoint
- **URL:** `POST /api/search/filters`
- **Body:** `{"semantic_class": "building", "change_type": "new", "min_area": 100}`

### C. CSV/JSON Export Endpoint
- **URL:** `GET /api/search/export?format=csv&semantic_class=road`

---

## 5. Important Scientific Qualification
> **Search Limitation:** The search engine queries **detected change regions** stored in the database. It does **not** perform open-ended object detection across raw unparsed satellite rasters. Queries state *"Change overlapping [Class]"* rather than unverified physical claims.
