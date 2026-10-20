# Video Demonstration Script & Timeline (3.5 Minutes)

---

### Timeline Breakdown

| Time Marker | Section Title | Visual Screen Content | Audio Voiceover Script |
| :---: | :--- | :--- | :--- |
| **0:00 – 0:20** | **Problem Statement** | Title Slide & Split Satellite Imagery | *"Satellite remote sensing generates massive multi-temporal imagery, but manual change detection is slow and raw difference maps fail to explain what changed."* |
| **0:20 – 0:40** | **Dataset & Constraints** | OSCD Dataset Slide & Band Grid | *"Using the OSCD benchmark dataset, our goal was to build an end-to-end change detection system operating on 100% CPU hardware without GPU reliance."* |
| **0:40 – 1:10** | **System Architecture** | Architecture Flowchart Diagram | *"We extract 32 spectral, temporal, texture, and edge features, feed them to a tuned XGBoost classifier, and intersect predictions with a spectral index semantic layer."* |
| **1:10 – 1:30** | **Launching Flask UI** | Browser opening `http://127.0.0.1:5000` | *"We launch our Flask web application, which provides both single-click benchmark testing and custom image uploads."* |
| **1:30 – 1:50** | **Running Analysis** | Clicking `Option A: test_006` scene | *"We select test scene 006 and click Analyze. In less than 0.4 seconds on CPU, full inference completes."* |
| **1:50 – 2:10** | **Results & Maps** | 6-Panel Comparison Grid | *"The system renders Before and After RGB crops, the XGBoost binary change map, the semantic mask, and the bounding box overlay."* |
| **2:10 – 2:30** | **Region Inspection** | Filtering Region Table & Region Detail Page | *"Below, the region table ranks changed objects by Significance Score. Clicking Inspect Region displays 4-panel crops and geometric metrics."* |
| **2:30 – 2:50** | **Validation Metrics** | Benchmark Comparison Chart | *"Evaluated on 20 unseen test scenes, XGBoost achieved a Mean F1 of 0.4606 and IoU of 0.3244, outperforming the baseline by 110%."* |
| **2:50 – 3:10** | **Limitations & Wording** | Limitations Accordion & Table | *"Our system maintains scientific honesty: outputs report 'Change overlapping road class' rather than unverified construction claims."* |
| **3:10 – 3:30** | **Conclusion & Wrap-Up** | Final Summary Slide & Code Repository | *"In conclusion, we have built a lightweight, explainable change detection prototype. Thank you for watching!"* |
