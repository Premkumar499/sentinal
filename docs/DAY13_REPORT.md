# Day 13 Technical Report: Explainable Change Analysis & Interpretability Layer

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Primary Model:** XGBoost Tuned Classifier (Threshold = 0.72, Frozen)  
**Explainability Layer:** `src/explanation.py` & `config/day13_config.yaml`  

---

## 1. Executive Summary & Objective

Day 13 built an **explainable change-analysis layer** that answers key analyst questions:
1. *Why was this region detected?*
2. *What changed between T1 and T2?*
3. *Which features contributed to the detection?*
4. *What semantic class overlaps the change?*
5. *How large is the region?*
6. *How strong is the model's change score?*
7. *Is the result uncertain?*
8. *What evidence supports the interpretation?*
9. *What should the user NOT conclude from this result?*

Zero model retraining or metric modifications were performed during Day 13. The XGBoost detector remains 100% frozen.

---

## 2. Configuration & System Freeze (`config/day13_config.yaml`)

- Locked down model version (`xgboost_v1.0_frozen`), decision threshold (`0.72`), feature list (`delta_NDVI`, `delta_NDBI`, `mean_spectral_diff`, etc.), region minimum area (`10 px`), and caching settings.

---

## 3. Deterministic Region Explanation Engine (`src/explanation.py`)

- **Feature Evidence Extraction:** Calculates $T_1$, $T_2$, and difference $\Delta$ values for NDVI, NDBI, NIR, and spectral channels.
- **Global Feature Importance:** Audits model feature importances from XGBoost booster gain (saved to `outputs/explanations/global_feature_importance.json`).
- **Quality & Uncertainty Check:** Evaluates `quality_status` (`PASS`, `LIMITED`, `UNCERTAIN`) based on region size, change score, and semantic class validity.
- **Deterministic Change Story:** Generates reproducible template-based human-readable explanations:
  - *"Before: Background ($T_1$). After: Road class region ($T_2$). The model's change score is 0.87."*
- **Cautious Disclaimer ("What NOT to conclude"):** *"This result indicates detected spectral change associated with a land-cover region. It does NOT independently confirm physical construction or demolition."*

---

## 4. Flask Model Explainability UI & REST APIs

- **`/model` Page (`app/templates/model_info.html`):** Interactive model audit page with spec sheet, global feature gain progress bars, OSCD evaluation metrics ($F_1 = 0.814$, $\text{IoU} = 0.687$, Precision = 83.2%, Recall = 79.7%), and scientific disclaimers.
- **Enhanced Region Details (`app/templates/region_detail.html`):** Includes the interactive **"Why Was This Region Selected?"** card, feature evidence table, change story, and export functionality.
- **REST Endpoints:**
  - `GET /api/model/features`
  - `GET /api/regions/<scene_id>/<region_id>/explanation`
  - `GET /api/regions/<scene_id>/<region_id>/export_explanation`

---

## 5. Verification & Testing

- **Unit Test Suite (`tests/test_explanation.py`):** Verified feature evidence extraction, global feature importance, explanation generation, reproducibility consistency across 3 repeated runs, and REST endpoints.
- **System Test Suite:** **55 / 55 pytest cases PASSED** cleanly.

---

## 6. Technical Conclusion

"Day 13 successfully implemented a lightweight, CPU-efficient, deterministic explainability engine for semantic satellite change detection. The engine provides transparent feature evidence, model feature gain rankings, quality status checks, and cautious scientific disclaimers, ensuring user trust without claiming unverified physical causality."
