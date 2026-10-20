import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Any

def get_placeholder_semantic_mask(image_shape: tuple, class_type: str = "road") -> np.ndarray:
    """
    Generate a placeholder semantic mask for architectural testing.
    0 = background, 1 = road, 2 = building, 3 = water, 4 = vegetation, 5 = bare_land
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.int32)
    if class_type == "road":
        mask[height//2 - 5 : height//2 + 5, :] = 1
    elif class_type == "building":
        mask[height//3 : height//3 + 20, width//3 : width//3 + 20] = 2
    elif class_type == "water":
        mask[:height//5, :width//5] = 3
    elif class_type == "vegetation":
        mask[height - height//4:, width - width//4:] = 4
    elif class_type == "bare_land":
        mask[height//2 : height//2 + 15, width//2 : width//2 + 15] = 5
    return mask

def get_semantic_mask(bands_norm: Dict[str, np.ndarray], source_type: str = "spectral_rules") -> np.ndarray:
    """
    Generate a categorical semantic land cover mask from normalized band dictionary.
    
    Classes:
    0 = Background / Unknown
    1 = Road (Narrow linear built-up features)
    2 = Building / Built-up (Wider built-up structures)
    3 = Water (High NDWI)
    4 = Vegetation (High NDVI)
    5 = Bare Land (High Red/SWIR, low NDVI/NDWI)
    """
    h, w = bands_norm["red"].shape
    if source_type == "placeholder":
        # Create a combined multi-class placeholder mask
        mask = np.zeros((h, w), dtype=np.int32)
        mask[h//2 - 5 : h//2 + 5, :] = 1 # road
        mask[h//3 : h//3 + 20, w//3 : w//3 + 20] = 2 # building
        mask[:h//5, :w//5] = 3 # water
        mask[h - h//4:, w - w//4:] = 4 # vegetation
        mask[h//2 : h//2 + 15, w//2 : w//2 + 15] = 5 # bare land
        return mask

    # Retrieve normalized band values (expected scale: 0.0 to 1.0)
    red = bands_norm["red"]
    green = bands_norm["green"]
    nir = bands_norm["nir"]
    swir = bands_norm["swir"]
    blue = bands_norm.get("blue", red)

    eps = 1e-6
    # Compute indices
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir - nir) / (swir + nir + eps)
    vari = (green - red) / (green + red - blue + eps)

    # Initialize mask
    semantic_mask = np.zeros((h, w), dtype=np.int32)

    # 1. Water detection: Primary optical signature is positive NDWI (water strongly absorbs NIR).
    # For RGB fallbacks, true water is dark (red < 0.22), with strong blue/cyan dominance (blue > red * 1.35 and green > red * 1.10),
    # low overall brightness (total < 0.70), and strictly positive NDWI. Bare land, shadows, and dry soil are excluded.
    water_mask = (ndwi > 0.12) | (
        (ndwi > 0.02) & (blue > red * 1.35) & (green > red * 1.10) & 
        (red < 0.22) & (red + green + blue < 0.70)
    )
    semantic_mask[water_mask] = 3

    # 2. Vegetation detection: High NDVI, positive VARI, or green reflectance dominance (excluding water)
    veg_mask = ((ndvi > 0.20) | (vari > 0.05) | ((green > red * 1.12) & (green > blue))) & (~water_mask)
    semantic_mask[veg_mask] = 4

    # 3. Built-up detection: NDBI > 0.05 (excluding water and vegetation)
    built_up_mask = (ndbi > 0.05) & (~water_mask) & (~veg_mask)
    
    if np.sum(built_up_mask) > 0:
        # Distance transform to differentiate roads (narrow/linear) from large buildings
        built_up_uint8 = (built_up_mask * 255).astype(np.uint8)
        dist = cv2.distanceTransform(built_up_uint8, cv2.DIST_L2, 5)
        
        # Heuristic: narrow features (thickness <= ~5 pixels/50m) are Roads (1)
        road_mask = built_up_mask & (dist > 0) & (dist <= 2.2)
        # Wider features are Buildings (2)
        building_mask = built_up_mask & (dist > 2.2)
        
        semantic_mask[building_mask] = 2
        semantic_mask[road_mask] = 1

    # 4. Bare Land detection: low vegetation, low water, low built-up, higher red band
    bare_mask = (semantic_mask == 0) & (ndvi <= 0.2) & (ndwi <= 0.0) & (ndbi <= 0.05) & (red > 0.12)
    semantic_mask[bare_mask] = 5

    return semantic_mask

def compute_semantic_change(change_mask: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    """
    Combine binary change mask and semantic mask to identify semantic-specific changes.
    """
    assert change_mask.shape == semantic_mask.shape, "Shape mismatch between change mask and semantic mask"
    # Wherever binary change occurs, keep the semantic class; otherwise, it remains 0 (background/no change)
    return (change_mask == 1) * semantic_mask

def compute_road_change(change_mask: np.ndarray, road_mask_t1: np.ndarray, road_mask_t2: np.ndarray = None) -> np.ndarray:
    """
    Compute road-specific change mask.
    If road_mask_t2 is provided, detects change overlapping either T1 or T2 roads.
    """
    if road_mask_t2 is None:
        return (change_mask == 1) & (road_mask_t1 == 1)
    else:
        road_union = (road_mask_t1 == 1) | (road_mask_t2 == 1)
        return (change_mask == 1) & road_union

def compute_temporal_semantic_change(change_mask: np.ndarray, t1_mask: np.ndarray, t2_mask: np.ndarray) -> np.ndarray:
    """
    Compute semantic change utilizing both T1 and T2 semantic masks.
    This resolves the exact transition category for changed pixels.
    """
    assert change_mask.shape == t1_mask.shape == t2_mask.shape
    
    # Initialize output map
    semantic_change = np.zeros_like(change_mask, dtype=np.int32)
    
    # Changed pixels
    changed_pixels = (change_mask == 1)
    
    # Rule: If change occurred, map the post-change (T2) semantic class.
    # If the T2 class is 0 (Background/Unknown), map it as 6 ("unresolved change").
    t2_classes = t2_mask[changed_pixels]
    mapped_classes = t2_classes.copy()
    mapped_classes[mapped_classes == 0] = 6 # Mark unclassified change as 6 (unknown change)
    
    semantic_change[changed_pixels] = mapped_classes
    return semantic_change

def extract_change_regions(semantic_change_mask: np.ndarray, min_size: int = 5) -> List[Dict[str, Any]]:
    """
    Extract object-level regions from the semantic change map using Connected Component Analysis.
    Returns a list of dictionaries with bounding box and centroid details.
    """
    regions = []
    unique_classes = np.unique(semantic_change_mask)
    # Exclude background/no-change (0)
    unique_classes = unique_classes[unique_classes != 0]
    
    region_id_counter = 1
    
    # Class mapping for names
    class_map = {1: "road", 2: "building", 3: "water", 4: "vegetation", 5: "bare_land", 6: "unknown"}

    for c in unique_classes:
        class_binary = (semantic_change_mask == c).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_binary)
        
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_size:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx = float(centroids[i][0])
                cy = float(centroids[i][1])
                
                regions.append({
                    "Region ID": region_id_counter,
                    "Class ID": int(c),
                    "Class": class_map.get(int(c), "unknown"),
                    "Area": area,
                    "X": x,
                    "Y": y,
                    "Width": w,
                    "Height": h,
                    "Centroid X": cx,
                    "Centroid Y": cy
                })
                region_id_counter += 1
                
    return regions

def determine_change_type(c1: int, c2: int) -> str:
    """
    Determine the qualitative change type based on T1 class (c1) and T2 class (c2).
    c1/c2 class mapping: 0=background/unknown, 1=road, 2=building, 3=water, 4=vegetation, 5=bare_land
    """
    if c1 == 0 and c2 != 0:
        return "new"
    elif c1 != 0 and c2 == 0:
        return "removed"
    elif c1 != 0 and c2 != 0:
        if c1 == c2:
            return "expanded"
        else:
            return "transition"
    else:
        return "unknown"

def calculate_semantic_transition_matrix(
    t1_mask: np.ndarray,
    t2_mask: np.ndarray,
    change_mask: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Calculate pixel transition matrix from T1 to T2 over changed pixels.
    """
    class_map = {0: "background", 1: "road", 2: "building", 3: "water", 4: "vegetation", 5: "bare_land", 6: "unknown"}
    changed = (change_mask == 1)
    t1_c = t1_mask[changed]
    t2_c = t2_mask[changed]

    matrix = []
    unique_pairs = {}
    for p1, p2 in zip(t1_c, t2_c):
        pair = (int(p1), int(p2))
        unique_pairs[pair] = unique_pairs.get(pair, 0) + 1

    for (p1, p2), count in sorted(unique_pairs.items(), key=lambda item: item[1], reverse=True):
        matrix.append({
            "from_class": class_map.get(p1, "unknown"),
            "to_class": class_map.get(p2, "unknown"),
            "from_id": p1,
            "to_id": p2,
            "pixel_count": count
        })
    return matrix

def extract_change_regions(
    semantic_change_mask: np.ndarray,
    min_size: int = 5,
    t1_mask: np.ndarray = None,
    t2_mask: np.ndarray = None,
    prob_map: np.ndarray = None,
    t1_norm: Dict[str, np.ndarray] = None,
    t2_norm: Dict[str, np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Extract object-level regions from the semantic change map using Connected Component Analysis.
    Enhanced for Day 8 with change types, geometric shape analysis, significance score, and quality flags.
    """
    regions = []
    unique_classes = np.unique(semantic_change_mask)
    unique_classes = unique_classes[unique_classes != 0]
    
    region_id_counter = 1
    class_map = {0: "unknown", 1: "road", 2: "building", 3: "water", 4: "vegetation", 5: "bare_land", 6: "unknown"}

    # Precompute index delta maps if normalized bands exist
    d_ndvi = None
    d_ndbi = None
    if t1_norm and t2_norm:
        eps = 1e-6
        r1, g1, nir1, swir1 = t1_norm["red"], t1_norm["green"], t1_norm["nir"], t1_norm["swir"]
        r2, g2, nir2, swir2 = t2_norm["red"], t2_norm["green"], t2_norm["nir"], t2_norm["swir"]

        ndvi1 = (nir1 - r1) / (nir1 + r1 + eps)
        ndvi2 = (nir2 - r2) / (nir2 + r2 + eps)
        d_ndvi = ndvi2 - ndvi1

        ndbi1 = (swir1 - nir1) / (swir1 + nir1 + eps)
        ndbi2 = (swir2 - nir2) / (swir2 + nir2 + eps)
        d_ndbi = ndbi2 - ndbi1

    for c in unique_classes:
        class_binary = (semantic_change_mask == c).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_binary)
        
        class_min_size = max(min_size, 35) if int(c) == 3 else min_size
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= class_min_size:
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx = float(centroids[i][0])
                cy = float(centroids[i][1])
                
                comp_mask = (labels == i)

                # Determine predominant T1 & T2 classes inside this region
                t1_c_id = 0
                t2_c_id = int(c)
                if t1_mask is not None and t2_mask is not None:
                    t1_vals = t1_mask[comp_mask]
                    t2_vals = t2_mask[comp_mask]
                    if len(t1_vals) > 0:
                        t1_c_id = int(np.bincount(t1_vals).argmax())
                    if len(t2_vals) > 0:
                        t2_c_id = int(np.bincount(t2_vals).argmax())

                change_type = determine_change_type(t1_c_id, t2_c_id)

                # Probabilities & Spectral Deltas
                p_mean = 0.85
                if prob_map is not None:
                    p_mean = float(np.mean(prob_map[comp_mask]))

                mean_d_ndvi = 0.0
                if d_ndvi is not None:
                    mean_d_ndvi = float(np.mean(d_ndvi[comp_mask]))

                mean_d_ndbi = 0.0
                if d_ndbi is not None:
                    mean_d_ndbi = float(np.mean(d_ndbi[comp_mask]))

                # Geometric Shape Analysis
                aspect_ratio = round(float(w / max(1, h)), 2)
                compactness = round(float(area / max(1, w * h)), 2)

                # Significance Score = (Area / 100.0) * Mean Change Probability
                significance_score = round(float((area / 100.0) * p_mean), 2)

                # Quality Flags
                quality_flags = {
                    "alignment_ok": True,
                    "semantic_available": bool(c != 6),
                    "small_region": bool(area < 20),
                    "low_change_confidence": bool(p_mean < 0.75)
                }

                class_name = class_map.get(int(c), "unknown")

                # Qualified interpretation wording
                qual_description = f"Change overlapping {class_name} class ({change_type})"

                regions.append({
                    "Region ID": region_id_counter,
                    "Class ID": int(c),
                    "Class": class_name,
                    "T1 Class": class_map.get(t1_c_id, "unknown"),
                    "T2 Class": class_map.get(t2_c_id, "unknown"),
                    "Change Type": change_type,
                    "Area": area,
                    "X": x,
                    "Y": y,
                    "Width": w,
                    "Height": h,
                    "Centroid X": cx,
                    "Centroid Y": cy,
                    "Aspect Ratio": aspect_ratio,
                    "Compactness": compactness,
                    "Mean Probability": round(p_mean, 4),
                    "NDVI Difference": round(mean_d_ndvi, 4),
                    "NDBI Difference": round(mean_d_ndbi, 4),
                    "Significance Score": significance_score,
                    "Quality Flags": quality_flags,
                    "Qualified Description": qual_description
                })
                region_id_counter += 1

    # Sort regions by Significance Score descending (Region Ranking)
    regions.sort(key=lambda r: r["Significance Score"], reverse=True)
    # Re-assign sequential rank IDs
    for rank_idx, reg in enumerate(regions, start=1):
        reg["Rank"] = rank_idx
        reg["Region ID"] = rank_idx

    return regions

def detect_semantic_change(
    t1_norm: Dict[str, np.ndarray],
    t2_norm: Dict[str, np.ndarray],
    change_mask: np.ndarray,
    semantic_source: str = "spectral_rules",
    prob_map: np.ndarray = None
) -> Dict[str, Any]:
    """
    Day 8 Enhanced Semantic Change Detection API.
    Combines the binary change mask with T1/T2 land cover masks, computes transition matrices,
    evaluates change types, shape properties, significance scores, and quality flags.
    """
    # 1. Generate semantic masks for both dates
    semantic_mask_t1 = get_semantic_mask(t1_norm, source_type=semantic_source)
    semantic_mask_t2 = get_semantic_mask(t2_norm, source_type=semantic_source)
    
    # 2. Combine change mask and semantic masks using temporal transitions
    semantic_change_mask = compute_temporal_semantic_change(change_mask, semantic_mask_t1, semantic_mask_t2)

    # 3. Calculate semantic transition matrix
    transition_matrix = calculate_semantic_transition_matrix(semantic_mask_t1, semantic_mask_t2, change_mask)
    
    # 4. Extract connected components / objects with Day 8 analytics
    min_size_filter = 10
    regions = extract_change_regions(
        semantic_change_mask=semantic_change_mask,
        min_size=min_size_filter,
        t1_mask=semantic_mask_t1,
        t2_mask=semantic_mask_t2,
        prob_map=prob_map,
        t1_norm=t1_norm,
        t2_norm=t2_norm
    )
    
    # 5. Compute statistics
    total_changed_pixels = int(np.sum(change_mask == 1))
    
    stats = {
        "total_changed_pixels": total_changed_pixels,
        "road_pixels": int(np.sum(semantic_change_mask == 1)),
        "building_pixels": int(np.sum(semantic_change_mask == 2)),
        "water_pixels": int(np.sum(semantic_change_mask == 3)),
        "vegetation_pixels": int(np.sum(semantic_change_mask == 4)),
        "bare_land_pixels": int(np.sum(semantic_change_mask == 5)),
        "unknown_pixels": int(np.sum(semantic_change_mask == 6)),
        "num_regions": len(regions),
        "largest_region": {}
    }
    
    if len(regions) > 0:
        largest = max(regions, key=lambda x: x["Area"])
        stats["largest_region"] = largest

    return {
        "change_mask": change_mask,
        "semantic_mask_t1": semantic_mask_t1,
        "semantic_mask_t2": semantic_mask_t2,
        "semantic_change_mask": semantic_change_mask,
        "transition_matrix": transition_matrix,
        "regions": regions,
        "statistics": stats
    }

