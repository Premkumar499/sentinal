import os
import sys
import time
import uuid
import json
import yaml
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

# Import core system services
from src.inference import InferenceEngine
from src.explanation import ExplanationEngine

class AnalysisError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")

def load_app_config(config_path: str = "config/app_config.yaml") -> Dict[str, Any]:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        cfg_file = Path("config/day13_config.yaml")
    
    if cfg_file.exists():
        with open(cfg_file, "r") as f:
            return yaml.safe_load(f)
    return {
        "paths": {
            "model_path": "models/xgboost_change_detector.json",
            "database_path": "data/change_regions.db",
            "runs_dir": "outputs/runs"
        },
        "inference": {"threshold": 0.72, "min_area_pixels": 10}
    }

def validate_app_config(config: Dict[str, Any]) -> bool:
    """Validate startup config and model existence."""
    model_path = config.get("paths", {}).get("model_path", "models/xgboost_change_detector.json")
    if not Path(model_path).exists():
        raise AnalysisError("MODEL_LOAD_ERROR", f"Configured model file does not exist: {model_path}")
    return True

class MasterPipeline:
    """Consolidated master pipeline for end-to-end change analysis."""

    def __init__(self, config_path: str = "config/app_config.yaml"):
        self.config = load_app_config(config_path)
        validate_app_config(self.config)

        self.model_path = self.config["paths"]["model_path"]
        self.threshold = self.config.get("inference", {}).get("threshold", 0.72)
        
        # Initialize sub-services
        self.engine = InferenceEngine(config_path="config.yaml")
        self.explanation_engine = ExplanationEngine()

    def validate_input(self, t1_img: Any, t2_img: Any) -> Tuple[Any, Any]:
        """Rigorous input validation for imagery."""
        if t1_img is None or t2_img is None:
            raise AnalysisError("INVALID_INPUT", "Input imagery cannot be None or empty.")

        if isinstance(t1_img, np.ndarray) and isinstance(t2_img, np.ndarray):
            if t1_img.size == 0 or t2_img.size == 0:
                raise AnalysisError("INVALID_INPUT", "Input image contains zero pixels.")

            if np.isnan(t1_img).any() or np.isnan(t2_img).any():
                raise AnalysisError("INVALID_INPUT", "Input image contains NaN values.")

            if np.isinf(t1_img).any() or np.isinf(t2_img).any():
                raise AnalysisError("INVALID_INPUT", "Input image contains Inf values.")

            if t1_img.shape[:2] != t2_img.shape[:2]:
                h, w = t1_img.shape[:2]
                t2_img = cv2.resize(t2_img, (w, h), interpolation=cv2.INTER_AREA)

        return t1_img, t2_img

    def run_change_analysis(
        self,
        t1_input: Union[str, np.ndarray],
        t2_input: Union[str, np.ndarray],
        scene_id: str = "scene_01",
        analysis_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Master pipeline execution function."""
        start_time = time.time()

        # Validate inputs
        self.validate_input(t1_input, t2_input)

        if not analysis_id:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            analysis_id = f"run_{timestamp}_{short_id}"

        runs_base = Path(self.config["paths"].get("runs_dir", "outputs/runs"))
        run_dir = runs_base / analysis_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # If inputs are numpy arrays, save temporary PNG files in run_dir
        if isinstance(t1_input, np.ndarray):
            t1_file = run_dir / "temp_t1.png"
            cv2.imwrite(str(t1_file), cv2.cvtColor(t1_input, cv2.COLOR_RGB2BGR))
            t1_path = str(t1_file)
        else:
            t1_path = str(t1_input)

        if isinstance(t2_input, np.ndarray):
            t2_file = run_dir / "temp_t2.png"
            cv2.imwrite(str(t2_file), cv2.cvtColor(t2_input, cv2.COLOR_RGB2BGR))
            t2_path = str(t2_file)
        else:
            t2_path = str(t2_input)

        # Run inference via engine
        try:
            results_json, regions, run_dir_path = self.engine.run_change_detection(
                t1_input=t1_path,
                t2_input=t2_path,
                threshold=self.threshold,
                run_id=analysis_id
            )
        except Exception as e:
            raise AnalysisError("FEATURE_ERROR", f"Change detection failed: {str(e)}")

        elapsed_seconds = round(time.time() - start_time, 3)

        manifest = {
            "status": "COMPLETED",
            "analysis_id": analysis_id,
            "scene_id": results_json.get("scene_id", scene_id),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": "xgboost_v1.0_frozen",
            "threshold": self.threshold,
            "changed_pixels": int(results_json.get("changed_pixels", 0)),
            "change_percentage": float(results_json.get("change_percentage", 0.0)),
            "region_count": len(regions),
            "processing_time_seconds": elapsed_seconds
        }

        with open(run_dir / "session_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "success",
            "analysis_id": analysis_id,
            "scene_id": results_json.get("scene_id", scene_id),
            "manifest": manifest,
            "regions": regions,
            "statistics": results_json.get("semantic_breakdown", {}),
            "run_dir": str(run_dir),
            "results": results_json
        }
