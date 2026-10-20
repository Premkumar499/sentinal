import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def get_geospatial_metadata(image_path: str) -> Dict[str, Any]:
    """Inspect raster file for geospatial metadata (CRS, transform, resolution, bounds)."""
    p = Path(image_path)
    meta = {
        "is_georeferenced": False,
        "crs": None,
        "transform": None,
        "resolution_m": None,
        "width": 0,
        "height": 0,
        "coordinate_type": "pixel"
    }

    if not p.exists():
        return meta

    if HAS_RASTERIO and p.suffix.lower() in [".tif", ".tiff"]:
        try:
            with rasterio.open(p) as src:
                meta["width"] = src.width
                meta["height"] = src.height
                if src.crs and src.transform and src.transform != rasterio.Affine.identity():
                    meta["is_georeferenced"] = True
                    meta["crs"] = str(src.crs)
                    meta["transform"] = list(src.transform)
                    meta["coordinate_type"] = "geographic"
                    
                    # Compute resolution in meters
                    res_x = abs(src.transform.a)
                    res_y = abs(src.transform.e)
                    meta["resolution_m"] = [round(res_x, 2), round(res_y, 2)]
                    meta["bounds"] = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        except Exception:
            pass

    # Fallback to image dimensions if not georeferenced or standard raster (PNG/JPG)
    if meta["width"] == 0 or meta["height"] == 0:
        try:
            img = cv2.imread(str(p))
            if img is not None:
                meta["height"], meta["width"] = img.shape[:2]
        except Exception:
            meta["width"], meta["height"] = 600, 600

    # Default Sentinel-2 OSCD 10m spatial resolution assumption for area calculations if unreferenced
    if meta["resolution_m"] is None:
        meta["resolution_m"] = [10.0, 10.0]

    return meta

def pixel_to_geo(x: float, y: float, metadata: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Convert pixel (X, Y) to (latitude, longitude) if valid geospatial transform exists."""
    if not metadata.get("is_georeferenced") or not metadata.get("transform"):
        return None

    if HAS_RASTERIO:
        try:
            transform = rasterio.Affine(*metadata["transform"])
            lon, lat = rasterio.transform.xy(transform, y, x)
            return round(lat, 6), round(lon, 6)
        except Exception:
            return None
    return None

def geo_to_pixel(lat: float, lon: float, metadata: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Convert (latitude, longitude) to pixel (X, Y) if valid geospatial transform exists."""
    if not metadata.get("is_georeferenced") or not metadata.get("transform"):
        return None

    if HAS_RASTERIO:
        try:
            transform = rasterio.Affine(*metadata["transform"])
            inv_transform = ~transform
            col, row = inv_transform * (lon, lat)
            return int(round(col)), int(round(row))
        except Exception:
            return None
    return None

def calculate_region_area_m2(area_pixels: int, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate real-world area in m2 and hectares if spatial resolution is available."""
    res = metadata.get("resolution_m", [10.0, 10.0])
    pixel_area_m2 = res[0] * res[1]
    area_m2 = round(area_pixels * pixel_area_m2, 2)
    area_ha = round(area_m2 / 10000.0, 4)

    return {
        "area_pixels": area_pixels,
        "pixel_resolution_m": res,
        "area_m2": area_m2,
        "area_hectares": area_ha,
        "is_exact_conversion": metadata.get("is_georeferenced", False)
    }

def generate_region_geojson(scene_id: str, regions: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate GeoJSON FeatureCollection if valid geospatial metadata exists."""
    if not metadata.get("is_georeferenced"):
        return None

    features = []
    for r in regions:
        bbox = r.get("Bounding Box [X,Y,W,H]", [r.get("X", 0), r.get("Y", 0), r.get("Width", 0), r.get("Height", 0)])
        x, y, w, h = bbox
        
        # Bbox corners in geo coords
        top_left = pixel_to_geo(x, y, metadata)
        top_right = pixel_to_geo(x + w, y, metadata)
        bottom_right = pixel_to_geo(x + w, y + h, metadata)
        bottom_left = pixel_to_geo(x, y + h, metadata)
        
        if top_left and bottom_right:
            polygon_coords = [[
                [top_left[1], top_left[0]],
                [top_right[1], top_right[0]],
                [bottom_right[1], bottom_right[0]],
                [bottom_left[1], bottom_left[0]],
                [top_left[1], top_left[0]]
            ]]
            
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords
                },
                "properties": {
                    "region_id": int(r.get("Region ID", r.get("id", 0))),
                    "scene_id": scene_id,
                    "semantic_class": str(r.get("Class", "unknown")),
                    "change_type": str(r.get("Change Type", "unknown")),
                    "area_pixels": int(r.get("Area", 0)),
                    "change_score": float(r.get("Significance Score", 0.0))
                }
            })

    if not features:
        return None

    return {
        "type": "FeatureCollection",
        "scene_id": scene_id,
        "crs": metadata.get("crs"),
        "features": features
    }

def draw_region_boundaries_image(rgb_image: np.ndarray, regions: List[Dict[str, Any]], output_path: str):
    """Draw bounding boxes, boundaries, and Region IDs over RGB image and save."""
    img_copy = rgb_image.copy()
    if img_copy.max() <= 1.0:
        img_copy = (img_copy * 255).astype(np.uint8)
    
    if len(img_copy.shape) == 2 or img_copy.shape[2] == 1:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    elif img_copy.shape[2] == 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

    for r in regions:
        bbox = r.get("Bounding Box [X,Y,W,H]", [r.get("X", 0), r.get("Y", 0), r.get("Width", 0), r.get("Height", 0)])
        x, y, w, h = [int(v) for v in bbox]
        r_id = int(r.get("Region ID", r.get("id", 0)))
        cls_name = str(r.get("Class", "region")).upper()

        # Color coding by class
        color = (0, 0, 255) # Red default
        if "road" in cls_name.lower(): color = (255, 0, 0) # Blue
        elif "building" in cls_name.lower(): color = (0, 0, 255) # Red
        elif "water" in cls_name.lower(): color = (255, 255, 0) # Cyan
        elif "vegetation" in cls_name.lower(): color = (0, 255, 0) # Green

        # Draw bounding rectangle
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        
        # Label text
        label = f"R{r_id}: {cls_name}"
        cv2.putText(img_copy, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_copy)
