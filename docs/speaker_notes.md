# Speaker Notes (Slide-by-Slide Presenter Script)

---

### Slide 1: Title Slide
> *"Welcome everyone. Today we present our AI/ML system for Semantic Satellite Change Detection. This project focuses on detecting surface changes between two satellite images and interpreting what physical land-cover class overlaps the changed area."*

### Slide 2: Problem Statement
> *"Detecting changes in satellite images is vital for monitoring urban growth and natural environments. However, traditional systems produce raw binary masks that tell you 'where' something changed, but fail to explain 'what' changed. Furthermore, deep learning solutions often require high-end GPUs unavailable in low-cost deployments."*

### Slide 3: Motivation & Real-World Use Cases
> *"Our goal was to build a system that bridges the gap between binary change detection and semantic explanation while adhering strictly to a 100% CPU compatibility constraint."*

### Slide 4: Core Project Objectives
> *"We achieved this by combining tabular feature engineering, a tuned XGBoost classifier, and a rule-based spectral semantic layer to extract bounded change regions and categorize them logically."*

### Slide 5: Dataset (OSCD)
> *"We trained and evaluated our system on the OSCD dataset—a benchmark of 100 aligned Sentinel-2 satellite image pairs at 10m spatial resolution across diverse global locations."*

### Slide 6: System Architecture
> *"Here you see our end-to-end pipeline: from raw multi-temporal imagery to 32 feature maps, XGBoost inference, spectral rule intersection, region extraction, and finally display on our Flask dashboard."*

### Slide 7: Feature Engineering
> *"We engineered 32 features including spectral band differences, band ratios, 3x3 local texture statistics, Sobel edge magnitudes, and built-up distance transform features."*

### Slide 8: XGBoost Model Selection & Tuning
> *"XGBoost was chosen because it excels at tabular pixel classification and executes full scene inference in under 0.40 seconds on standard CPU processors. We optimized the probability threshold to 0.72."*

### Slide 9: Semantic Layer & Qualitative Change Types
> *"Our semantic layer uses spectral index rules like NDVI, NDWI, and NDBI along with shape thinness heuristics to categorize changes into Road, Building, Water, Vegetation, Bare Land, or Unknown, and assigns qualitative change types like New, Expanded, or Removed."*

### Slide 10: Quantitative Validation Results
> *"On 20 unseen test scenes evaluated under zero test-set tuning, XGBoost achieved a Mean F1-score of 0.4606 and IoU of 0.3244—a 110.9% improvement over the simple difference baseline."*

### Slide 11: Visual Results
> *"This 6-panel grid shows how the system progresses from raw satellite crops to binary change maps, color-coded semantic masks, and region bounding box overlays."*

### Slide 12: Error Analysis
> *"Honest scientific error analysis reveals that false positives stem mainly from seasonal crop shifts and solar shadow shifts, while false negatives occur primarily along small structures below the 10m resolution limit."*

### Slide 13: Web Application & REST API
> *"Our Flask web application allows users to upload custom image pairs or run sample benchmark scenes in one click, view filterable region tables, inspect 4-panel region crops, or access results programmatically via REST API."*

### Slide 14: System Limitations & Scientific Honesty
> *"To maintain scientific integrity, our system states 'Change overlapping road class' rather than unverified claims like 'Confirmed road construction'. It represents an explainable prototype, not a replacement for GIS ground verification."*

### Slide 15: Conclusion & Future Work
> *"In conclusion, we have built a CPU-friendly, explainable change detection system. Future work will explore deep learning Siamese architectures when GPU hardware is available. Thank you!"*
