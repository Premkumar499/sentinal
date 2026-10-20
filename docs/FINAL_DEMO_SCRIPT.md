# Live Demo Script (3–5 Minute Presentation Walkthrough)

**Project Name:** Semantic Satellite Change Detection  
**Target Audience:** Project Evaluators, Faculty Judges, Technical Reviewers  

---

## ⏱️ Step-by-Step Presentation Script

### 0:00 – 0:30 | Step 1: Introduction & Problem Context
> *"Good morning. Today we present our AI/ML-based **Semantic Satellite Change Detection System**. The objective of this project is to compare multi-temporal satellite imagery captured at two dates—$T_1$ and $T_2$—detect significant spatial changes, and interpret what land-cover class overlaps the changed area."*

### 0:30 – 1:00 | Step 2: System Architecture & CPU Constraints
> *"Our system is designed to run efficiently on 100% CPU hardware without requiring expensive GPU infrastructure. We extract 32 tabular spectral and spatial features from Sentinel-2 imagery, feed them to a tuned **XGBoost classifier** operating at threshold $0.72$, and intersect the predictions with a spectral index semantic reference layer."*

### 1:00 – 1:45 | Step 3: Executing Live Analysis
> *"Let us launch the live Flask application. We can upload custom bi-temporal satellite pairs or click **Option A** to analyze a benchmark scene from the OSCD dataset. We select **`test_006`** (or navigate to `/demo`) and click **🚀 Analyze Selected Sample Scene**."*

### 1:45 – 2:30 | Step 4: Reviewing Results Grid & Change Maps
> *"In less than $0.4$ seconds, the system generates a 6-panel comparison grid:
> 1. **Before ($T_1$) & After ($T_2$) RGB PNGs**
> 2. **XGBoost Binary Change Map:** Highlights candidate change pixels.
> 3. **Color-Coded Semantic Map:** Displays $T_2$ land-cover categories (Roads, Buildings, Water, Vegetation, Bare Land).
> 4. **Semantic Change Map & Overlay:** Visualizes exactly which land-cover class overlaps each change region with bounding boxes."*

### 2:30 – 3:15 | Step 5: Object Region Details & Significance Ranking
> *"Below the visual maps, the system displays the **Significant Changed Regions Table**. We can filter regions by Class or Change Type (`New`, `Expanded`, `Removed`, `Transition`). 
> Clicking **Inspect Region R1** brings up the **Region Detail Inspection Page**, displaying 4-panel crops, bounding box metrics, spectral deltas ($\Delta\text{NDVI}$, $\Delta\text{NDBI}$), and automated quality flags."*

### 3:15 – 4:00 | Step 6: Benchmark Metrics & Scientific Honesty
> *"On our 20 unseen OSCD test scenes, XGBoost achieved a **Mean F1-score of 0.4606** and **IoU of 0.3244**, outperforming the simple difference baseline by **+110.9%**."*
> *"Crucially, our system maintains scientific honesty: outputs report **'Change overlapping road class'** rather than unverified claims like 'Confirmed construction'. This distinction preserves prototype integrity."*

---

## 💡 Fallback Demo Plan (If Network/Browser Fails)
If a browser issue occurs during live demo, navigate to `/demo?scene=test_006` or open `outputs/final_evaluation/final_report.html` for pre-rendered evaluation outputs.
