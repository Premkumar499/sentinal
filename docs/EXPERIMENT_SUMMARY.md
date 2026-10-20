# Experiment Progression & Metrics Summary

## 1. Pipeline Evolution

| Phase | Description | Key Metric | Key Improvement |
| :--- | :--- | :--- | :--- |
| **Day 1 Baseline** | Simple Image Difference Map | Baseline Visual Map | Raw reflectance scaling & spectral difference |
| **Day 2 Model** | Untuned XGBoost Classifier | F1 = 0.4678, IoU = 0.3053 | 32 engineered features + class imbalance weighting |
| **Day 3 Model** | Tuned XGBoost Classifier ($threshold = 0.72$) | F1 = 0.5351, IoU = 0.3653 | Parameter tuning, morphological opening ($kernel=0$) |
| **Day 4 Layer** | Spectral Rule Semantic Classification | 5 Semantic Classes | Road thinness distance transform & spectral index rules |
| **Day 5 Prototype** | End-to-End Flask Web App | Runtime = 0.3262s | Interactive 6-panel dashboard, CCA region tables, REST API |
| **Day 6 Validation** | System Hardening & Auditing | 21/21 Tests Passed | Verified CLI/Flask 100% equivalence, zero memory leak |

---

## 2. Quantitative Performance Summary

| Metric | Day 2 Baseline | Day 3 Tuned Model | Day 6 Hardened System |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.4412 | 0.5336 | 0.5336 |
| **Recall** | 0.4978 | 0.5366 | 0.5366 |
| **F1 Score** | 0.4678 | **0.5351** | **0.5351** |
| **IoU (Jaccard)** | 0.3053 | **0.3653** | **0.3653** |
| **Inference Time (CPU)** | 0.1200s | 0.0850s | **0.0928s** |
