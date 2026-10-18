"""
Configuration settings for LucidPlus OCR backend.
Centralizes paths, model configurations, and detection/recognition thresholds.
"""

from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
IMAGES_DIR = (BASE_DIR / "images") if (BASE_DIR / "images").exists() else (SAMPLES_DIR / "input")


class ModelSettings(BaseModel):
    # ONNX Model Paths
    det_model_path: Path = MODELS_DIR / "det.onnx"
    rec_latin_model_path: Path = MODELS_DIR / "rec_latin.onnx"
    dict_latin_path: Path = MODELS_DIR / "dict_latin.txt"
    rec_greek_model_path: Path = MODELS_DIR / "rec_greek.onnx"
    dict_greek_path: Path = MODELS_DIR / "dict_greek.txt"

    # CPU Execution Settings
    intra_op_num_threads: int = 4
    inter_op_num_threads: int = 2

    # Detection Thresholds (DBNet)
    det_limit_side_len: int = 960
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    det_db_unclip_ratio: float = 1.6
    min_box_size: int = 3

    # Recognition Settings (SVTR)
    rec_image_height: int = 48
    rec_batch_size: int = 16
    rec_score_thresh: float = 0.3


class AppSettings(BaseModel):
    app_name: str = "OCR Engine API"
    app_version: str = "1.0.0"
    debug: bool = False
    cors_origins: list[str] = ["*"]
    models: ModelSettings = ModelSettings()


settings = AppSettings()
