# Day 9 Technical Report: Final Scientific Validation & Comprehensive Performance Evaluation

**Project Name:** Semantic Satellite Change Detection  
**Dataset:** OSCD (Onera Satellite Change Detection)  
**Primary ML Model:** XGBoost Tuned Classifier (Threshold = 0.72, Frozen)  
**Semantic Layer:** Spectral Index Rules (NDVI/NDWI/NDBI) + Distance Transform Heuristics  
**Execution Environment:** 100% CPU Compatible  

---

## 1. Executive Summary & Objective

Day 9 completed the **Final Scientific Validation** of the semantic satellite change-detection system. The objective was to systematically answer key research and performance questions: *How well does XGBoost detect change? How does it compare with the simple difference baseline? Which scenes work well vs fail? What error categories occur? Are semantic region interpretations consistent? What can we confidently claim vs NOT claim?*

Zero hyperparameter modifications or test-set tuning were performed during Day 9.

---

## 2. Frozen Experiment & Evaluation Protocol

The system was frozen prior to evaluation (`config/final_experiment.json` & `config/final_config.yaml`):
- **Model:** Tuned XGBoost Classifier (Random Seed: 42)
- **Features:** 32 Tabular Features (Spectral diffs, ratios, Sobel edge magnitude, 3x3 local mean/std, distance transform)
- **Probability Threshold:** $0.72$ (Class-weight balanced)
- **Post-Processing:** Morphological $3 \times 3$ close + Minimum component area $\ge 3$ pixels
- **Test Set:** 20 unseen OSCD test scenes (`test_000` through `test_019`).

---

## 3. Overall Benchmark Metrics

| Benchmark Metric | Mean Value | Standard Deviation | Median Value |
| :--- | :---: | :---: | :---: |
| **F1-Score** | **0.4606** | $\pm 0.2125$ | **0.4591** |
| **IoU Score (Jaccard)** | **0.3244** | $\pm 0.1835$ | **0.3004** |
| **Precision** | **0.5621** | $\pm 0.1699$ | **0.5803** |
| **Recall** | **0.4635** | $\pm 0.2833$ | **0.3881** |
| **Overall Accuracy** | **0.9575** | $\pm 0.0321$ | **0.9654** |
| **CPU Latency / Scene** | **0.38 sec** | $\pm 0.04 \text{ sec}$ | **0.37 sec** |

---

## 4. Scene-by-Scene Performance Ranking

### Top 3 Best Performing Scenes:
1. **`test_006`:** F1 = **0.7791** | IoU = **0.6382** (High contrast built-up surface expansion)
2. **`test_012`:** F1 = **0.7538** | IoU = **0.6049** (Distinct infrastructure boundaries)
3. **`test_010`:** F1 = **0.6961** | IoU = **0.5338** (Clear ground-clearing activity)

### Top 3 Worst Performing Scenes:
1. **`test_001`:** F1 = **0.1098** | IoU = **0.0581** (High seasonal crop vegetation variance)
2. **`test_007`:** F1 = **0.1375** | IoU = **0.0738** (Low spectral contrast bare land shift)
3. **`test_015`:** F1 = **0.1931** | IoU = **0.1069** (Solar angle shadow artifact)

---

## 5. Model Comparisons (Baseline & Evolutionary Metrics)

### A. XGBoost vs Simple Difference Baseline (Day 1)
| Model Version | Precision | Recall | F1-Score | IoU Score |
| :--- | :---: | :---: | :---: | :---: |
| **Simple Difference Baseline (Day 1)** | 0.2840 | 0.3120 | 0.2184 | 0.1265 |
| **XGBoost Tuned (Day 3-9)** | **0.5621** | **0.4635** | **0.4606** | **0.3244** |

*Result:* XGBoost achieved a **+110.9% improvement in F1-score** and **+156.4% improvement in IoU** over the simple difference baseline.

### B. Evolutionary Improvement (Day 2 vs Day 3-9)
| Development Stage | Feature Count | F1-Score | IoU Score | Key Enhancement |
| :--- | :---: | :---: | :---: | :--- |
| **Day 2 Baseline XGBoost** | 12 Bands/Diffs | 0.3412 | 0.2110 | Default hyperparameters |
| **Day 3-9 Tuned XGBoost** | 32 Features | **0.4606** | **0.3244** | Class-weight balancing, threshold 0.72, Sobel/texture features |

---

## 6. Error Analysis & Categorization

### A. False Positive Categories (`false_positive_categories.csv`)
1. **Seasonal Vegetation / Crop Shifts (38.2%):** Agricultural growth/harvest between satellite passes.
2. **Illumination & Solar Angle Variation (25.5%):** Solar azimuth differences causing brightness shifts.
3. **Shadow Shift along Tall Structures (16.4%):** Moving shadow contours along building edges.
4. **Registration Misalignment (10.9%):** Sub-pixel geometric offsets ($< 1\text{px}$).
5. **Unknown Noise (9.0%):** Ambient atmospheric reflectance shifts.

### B. False Negative Categories (`false_negative_categories.csv`)
1. **Small Object Boundary Cutoff (41.2%):** Changed structures $< 10\text{m}$ truncated by resolution limits.
2. **Low Spectral Contrast Materials (25.9%):** Built-up asphalt blending with surrounding soil.
3. **Mixed Pixel Boundary Attenuation (17.6%):** Edge pixels falling below threshold $0.72$.

---

## 7. What the System CAN vs CANNOT Claim

- **PERMITTED CLAIMS:**
  - *"Detected change overlaps a road-class region."*
  - *"Building-overlapping change detected."*
  - *"Water-associated spectral shift observed."*
- **PROHIBITED CLAIMS:**
  - ❌ *"New road constructed."*
  - ❌ *"Building demolished / erected."*
  - ❌ *"Flood confirmed."*

---

## 8. Reproducibility & Day 10 Plan

- Single-command reproducibility verified: `python scripts/run_final_evaluation.py`.
- **Day 10 Plan:** Final project packaging, presentation slide deck compilation, live demonstration rehearsal, and final submission archive generation.

---

## 9. Final Technical Conclusion

"Day 9 completed the final scientific validation of the semantic satellite change-detection system. The frozen XGBoost model was evaluated across the designated unseen test scenes using precision, recall, F1-score, IoU, confusion matrices, and qualitative error analysis. Results were compared with the original difference-based baseline. Semantic change outputs were evaluated separately and interpreted conservatively according to the available semantic/reference information. The final analysis identifies both the system's capabilities and its limitations without overstating unsupported real-world conclusions."
