# 🛰️ Smart India Hackathon (SIH) — Final Project Presentation & Demo Guide

**Project Title:** Semantic Satellite Change Detection, Multimodal AI Search & Environmental Audit Intelligence Platform  
**Target Platform:** Linux/Windows/macOS (100% CPU Compatible, Zero GPU Requirement)  
**Version:** `v1.2.0-final`  
**Master Entry Point:** `python run_project.py`

---

## 🎯 1. The 30-Second Elevator Pitch

> "Traditional satellite change detection outputs only noisy binary black-and-white masks, leaving domain analysts to manually inspect vast geographic areas.
>
> We built a **lightweight, CPU-only Earth Observation Intelligence Platform** that:
> 1. Detects physical surface changes in **< 0.12 seconds** using a tuned 32-feature XGBoost classifier.
> 2. Georeferences detected change regions across **38 Tamil Nadu districts** with true GPS coordinates and physical ground area ($m^2$).
> 3. Enables **Multimodal AI Vector Search** where officials can search satellite imagery using plain text (*'Find water reservoirs near urban areas'*) or upload an image to find visually similar regions across the state.
> 4. Serves an **Interactive GIS Satellite Map** and generates **1-click Official Environmental Audit Briefings (PDF)** with automated lakebed encroachment alerts and field verification checklists."

---

## 🏛️ 2. High-Level System Architecture

```
                                  SATELLITE IMAGERY
                       (Sentinel-2 / 38 Tamil Nadu Districts)
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  PHYSICAL ENGINE                MULTIMODAL SEARCH                   GIS ENGINE
 (32-Feature XGBoost            (Dual 512-d CLIP &              (Leaflet.js + Esri
  Change Classifier)              FAISS Vector Store)           Satellite + Footprints)
        │                                │                                │
        ▼                                ▼                                ▼
  Change Contours &                Text & Image-to-              Interactive GPS Map
  Pixel Probability                Satellite Area                with Area-Scaled
  (696+ GPS Regions)               (1,221 Vectors)               Pulsing Markers
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         ▼
                     GOVERNMENT ENVIRONMENTAL AUDIT ENGINE
                            (src/reporting.py)
            - Automated Threat Triage (Critical Lakebed Encroachments)
            - Hectares & Class Transition Depletion Breakdown
            - Actionable Ground Verification Checklist
                                         │
                                         ▼
                     OFFICIAL BRIEFING DASHBOARD (/reports)
                          (1-Click Print / PDF Export)
```

---

## 🌐 3. Subsystems & Navigation Table

| Subsystem | Live Route | Target Audience | Key Feature |
| :--- | :--- | :--- | :--- |
| **🏠 Change Detection Dashboard** | `http://127.0.0.1:5000/` | Remote Sensing Analysts | 6-Panel bi-temporal comparison ($T_1$, $T_2$, Mask, Probability, Semantic Map, Overlay) |
| **🗺️ Interactive GIS Satellite Map** | `http://127.0.0.1:5000/map` | GIS Specialists & Planners | Esri Satellite base-layer, GPS change markers scaled by area ($m^2$), district flight shortcuts |
| **🧠 Multimodal AI Vector Search** | `http://127.0.0.1:5000/semantic-search` | Search & Intelligence Teams | Plain-text to satellite area search & image-to-similar-area retrieval using 512-d FAISS index |
| **📄 Environmental Audit Reports** | `http://127.0.0.1:5000/reports` | District Collectors & TWAD | Automated threat triage, lakebed encroachment alerts, 1-click printable official PDF |
| **🔍 SQL & NL Query Engine** | `http://127.0.0.1:5000/search` | Database Analysts | Structured SQL queries, area thresholds, and transition filtering |
| **📊 Model Architecture & Audit** | `http://127.0.0.1:5000/model` | ML Evaluators | 32-feature global importance rankings and frozen threshold verification |

---

## 🎬 4. Step-by-Step 4-Minute Live Demo Script for Judges

### Minute 0:00 – 1:00: Main Dashboard & XGBoost Change Detection
1. Open `http://127.0.0.1:5000/`.
2. Select **`CHENNAI_CLOSE_Z14_R01_C02`** from the sample scene dropdown and click **"Run Change Detection"**.
3. **Point out to judges:**
   - Speed: Sub-second inference ($< 0.40$s) entirely on standard CPU.
   - The 6-panel glassmorphic comparison view ($T_1$ historical imagery, $T_2$ current imagery, XGBoost probability heatmap, morphological change mask, and semantic land-cover transition map).
   - The ranked region table below showing extracted bounding boxes, centroid coordinates, and ground area in $m^2$.

### Minute 1:00 – 2:00: Interactive GIS Satellite Map
1. Click **"🗺️ GIS Map"** in the top navbar (`http://127.0.0.1:5000/map`).
2. Click the **"🏙️ Chennai"** shortcut button to fly smoothly to Chennai.
3. Switch the base-layer toggle between **🛰️ Esri Satellite** and **🌑 Dark Matter**.
4. Click any pulsing change marker on the map:
   - Show how the right sidebar instantly synchronizes and displays the **4-panel visual comparison crop**.
   - Point out real-world metrics: Ground Area in $m^2$, Significance Score, and GPS coordinates ($13.2086^\circ\text{N}, 80.1589^\circ\text{E}$).

### Minute 2:00 – 3:00: Multimodal AI Vector Search
1. Click **"🧠 AI Multimodal Search"** (`http://127.0.0.1:5000/semantic-search`).
2. Click the prompt chip: **💧 Urban Reservoirs** (*"Find large water bodies and reservoirs near urban development"*).
3. Click **"⚡ Search"**:
   - Show instant sub-second retrieval across 1,221 indexed satellite vectors.
   - Highlight the **97.8% match badge**, district tags, and thumbnail preview cards.
4. Switch to **"🖼️ Image $\to$ Similar Area Search"** tab:
   - Click **"💧 Chennai Water Body Transition"** sample button to demonstrate reverse-image search retrieving identical geographic patterns across other Tamil Nadu districts.

### Minute 3:00 – 4:00: Government Environmental Audit & PDF Export
1. Click **"📄 Audit Reports"** (`http://127.0.0.1:5000/reports`).
2. Select **"Chennai"** or **"Coimbatore"**:
   - Show the **Official Government of Tamil Nadu Header**.
   - Show the automated Threat Triage badges: **🔴 Critical Lakebed Encroachment** (water bodies converted to built-up land).
   - Show the **Field Verification Checklist** with GPS targets and assigned departments (*Water Resources Department / TWAD*, *Forest Division*).
3. Click **"🖨️ Export / Print Official PDF"**:
   - Demonstrate the clean, high-contrast, publication-grade printable briefing page with sign-off lines for the District Collector.

---

## 💡 5. Answers to Tough Viva / Evaluation Questions

**Q: Why use XGBoost instead of a Deep Learning model (like ResNet or SegNet) for change detection?**  
> *"Deep learning architectures require dedicated GPU clusters, large power supplies, and high memory footprints ($>4\text{ GB}$). In disaster response and remote district collectorates, edge hardware is often limited to standard dual-core or quad-core laptops. Our 32-feature XGBoost pipeline delivers a **Mean F1 of 0.4606** (+110.9% over difference baselines) in **0.11 seconds on standard CPU**, requiring zero GPU infrastructure."*

**Q: How do your Text-to-Satellite and Image-to-Satellite search features work?**  
> *"We map both natural language text queries and satellite visual crops into a shared **512-dimensional normalized unit vector space** ($\mathbb{R}^{512}$) using a dual-encoder architecture. When an official searches for 'reservoir near urban areas', the text vector is compared against 1,221 indexed satellite vectors using **FAISS Inner Product (Cosine Similarity)**, followed by spatial and district post-filtering. This runs in $<0.04$ seconds on CPU."*

**Q: How do you prevent false positive changes caused by clouds, seasonal shadows, or sun angles?**  
> *"We employ three defense mechanisms:  
> 1. **Probability thresholding:** Our decision boundary is strictly frozen at $0.72$ (filtering out faint shadow shifts).  
> 2. **Morphological post-processing:** $3\times3$ morphological closing consolidates real physical structures while eliminating isolated salt-and-pepper noise.  
> 3. **Connected Component Analysis (CCA):** Regions smaller than 10–12 pixels are automatically suppressed."*

**Q: What is the real-world value for the Government of Tamil Nadu?**  
> *"For bodies like the Tamil Nadu Water Resources Department (TWAD), CMDA, and Forest Department, unauthorized encroachment on lakebeds and green belts is a multi-crore crisis. Our system automatically categorizes changes by threat severity, generates exact GPS inspection coordinates, and prints an actionable legal verification checklist for ground revenue inspectors in one click."*
