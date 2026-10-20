# Day 11 Technical Report: Semantic Change Searching, Indexing & Query Engine

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Primary Model:** XGBoost Tuned Classifier (Threshold = 0.72, Frozen)  
**Search Index:** SQLite (`data/change_regions.db`, 580 regions indexed)  

---

## 1. Executive Summary & Objective

Day 11 extended the Semantic Satellite Change Detection system from passive visualization to **interactive semantic change searching and natural language query interpretation**. Analysts can query detected regions using natural phrases (e.g. *"Show road changes"*, *"Find building changes"*, *"Show large changes"*, *"Find new road regions"*). 

Zero hyperparameter modifications or model retraining were performed during Day 11. Day 11 operates purely on top of existing frozen XGBoost inference outputs.

---

## 2. Region Data Model & SQLite Database (`data/change_regions.db`)

All processed regions across evaluation runs were indexed into SQLite database `data/change_regions.db` in table `change_regions`:
- **Indexed Fields:** `id`, `run_id`, `scene_id`, `region_id`, `semantic_class`, `change_type`, `area_pixels`, `change_score`, `centroid_x`, `centroid_y`, `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`, `t1_class`, `t2_class`.
- **Performance Indexes:** `idx_semantic_class`, `idx_change_type`, `idx_scene_id`, `idx_run_id`.
- **Query Latency:** $< 0.005$ seconds per search query (does not re-run XGBoost).

---

## 3. Natural Language Query Parser (`QueryParser`)

The rule-based query parser converts search text into structured intents:
- **Class Mapping:** `"road"`, `"building"`, `"water"`, `"vegetation"`, `"bare_land"`, `"unknown"`.
- **Change Type Mapping:** `"new"`, `"removed"`, `"expanded"`, `"transition"`, `"unknown"`.
- **Area & Score Modifiers:** `"large"`, `"small"`, `"high confidence"`.
- **Unsupported Class Handling:** Terms like `"airport"`, `"bridge"`, `"river"` return helpful status messages detailing supported semantic classes.

---

## 4. Flask Search UI & REST APIs

- **UI Page (`/search`):** Search input, quick filter buttons, interpreted query intent breakdown, search summary stats, interactive region cards, direct 4-panel detail view links (`/region/<run_id>/<region_id>`).
- **REST Endpoints:** `POST /api/search`, `POST /api/search/filters`, `GET /api/search/export`.

---

## 5. Testing & Verification

- **Unit Test Suite (`tests/test_day11_search.py`):** Verified query parser patterns, SQLite index creation, filtering, unsupported class handling, and API endpoints.
- **Overall System Test Suite:** **45 / 45 pytest cases PASSED** cleanly.

---

## 6. Conclusion

"Day 11 successfully implemented a lightweight, fast, SQLite-indexed semantic change search engine and query parser. The search engine allows users to rapidly filter detected land-cover change regions by class, change type, area, and confidence, fulfilling the project's goal of explainable semantic change interpretation."
