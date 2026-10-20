# System Limitations & Real-World Interpretation Rules

**Project Name:** Semantic Satellite Change Detection  
**Evaluation Status:** Scientifically Validated (Day 9)  

---

## 1. What the System CANNOT Do

To maintain scientific integrity, the following limitations are explicitly declared:

1. **Cannot Prove Construction or Demolition:** The system detects spectral and spatial changes between dates. It **cannot** confirm real-world physical events (e.g., active construction works, building demolition, or structural damage) without ground verification.
2. **Cannot Guarantee Automatic Object Extraction:** Spectral rules and thinness heuristics provide reference class associations, but do not replace high-resolution semantic segmentation neural networks.
3. **Cannot Confirm Natural Disasters or Floods:** Water-associated spectral changes indicate NDWI variance; they **do not confirm** flood events.
4. **Resolution Boundaries:** Sentinel-2 10m spatial resolution limits the detection of narrow infrastructure ($< 10\text{m}$ width) or individual small housing structures.
5. **Sensitivity to Illumination and Seasonality:** Seasonal vegetation shifts (crop growth/harvest) and solar illumination/shadow differences represent $63.7\%$ of total false positives.

---

## 2. Real-World Interpretation Rules

To prevent overstating model claims, all UI labels, reports, and API outputs adhere strictly to the following wording rules:

| Model Output Signal | Permitted Qualified Interpretation | Prohibited Claim |
| :--- | :--- | :--- |
| `Change = YES`, `Semantic = Road` | *"Detected change overlaps a road-class region."* | ❌ *"New road constructed."* |
| `Change = YES`, `Semantic = Building` | *"Building-overlapping change detected."* | ❌ *"Building demolished / erected."* |
| `Change = YES`, `Semantic = Water` | *"Water-associated change detected."* | ❌ *"Flood confirmed."* |
| `Change = YES`, `Semantic = Vegetation`| *"Vegetation-overlapping change detected."* | ❌ *"Deforestation confirmed."* |

---

## 3. Dataset Constraints (OSCD)

- The OSCD (Onera Satellite Change Detection) dataset contains **binary change annotations only** (0 = No Change, 1 = Change).
- Semantic land-cover classes are derived via rule-based spectral indices (NDVI/NDWI/NDBI) and morphological distance transforms rather than dense human-annotated semantic maps.
