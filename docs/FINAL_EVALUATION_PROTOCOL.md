# Final Scientific Evaluation Protocol

**Project Name:** Semantic Satellite Change Detection  
**Evaluation Protocol Version:** 1.0.0 (Frozen)  

---

## 1. Dataset Partitioning & Strict Separation

To ensure unbiased evaluation, the 100 OSCD scenes were strictly partitioned into three mutually exclusive sets:

- **Train Set (60 Scenes):** `train_000` through `train_059`. Used exclusively for training the XGBoost change detection model during Days 2 and 3.
- **Validation Set (20 Scenes):** `val_000` through `val_019`. Used exclusively for hyperparameter tuning, feature selection (selecting 32 spectral/spatial features), and probability threshold optimization ($0.72$).
- **Test Set (20 Scenes):** `test_000` through `test_019`. **Held out completely until Day 9.** Used exclusively for final scientific evaluation.

---

## 2. Freeze Constraint Policy

1. **Zero Parameter Tuning on Test Set:** The XGBoost model parameters, feature set, probability threshold ($0.72$), post-processing parameters (morphological $3 \times 3$ close + minimum component size $\ge 3$ pixels), and semantic spectral rules were frozen prior to evaluating the test set.
2. **Deterministic Inference:** All evaluation runs use fixed random seed `42` and strict deterministic feature computation pipelines.

---

## 3. Primary Metrics & Formulas

For each test scene, the predicted binary change mask is compared against the ground truth change mask pixel by pixel:

- **True Positives ($TP$):** Changed pixels correctly predicted as change.
- **True Negatives ($TN$):** Unchanged pixels correctly predicted as no change.
- **False Positives ($FP$):** Unchanged pixels incorrectly predicted as change.
- **False Negatives ($FN$):** Changed pixels incorrectly predicted as no change.

### Metric Definitions:
1. **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
2. **Precision:** $\frac{TP}{TP + FP + \epsilon}$
3. **Recall:** $\frac{TP}{TP + FN + \epsilon}$
4. **F1-Score:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon}$
5. **Intersection over Union (IoU / Jaccard):** $\frac{TP}{TP + FP + FN + \epsilon}$

---

## 4. Evaluation Outputs

The final evaluation script generates all deliverables in `outputs/final_evaluation/`:
- `scene_metrics.csv`
- `model_comparison.csv`
- `error_distribution.csv`
- `false_positive_categories.csv`
- `false_negative_categories.csv`
- `confusion_matrix.png`
- `probability_distribution.png`
- `threshold_sensitivity.png`
- `final_results.csv`
- `final_report.html`
- `error_gallery/`
