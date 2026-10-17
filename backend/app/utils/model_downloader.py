1"""
Model Downloader Utility.
Downloads ONNX models and character dictionaries for detection and multilingual recognition
from the HuggingFace repository (monkt/paddleocr-onnx).
"""

import os
import sys
import logging
import requests
from pathlib import Path
from typing import Dict

logger = logging.getLogger("lucid_ocr.downloader")

HF_BASE_URL = "https://huggingface.co/monkt/paddleocr-onnx/resolve/main"

MODELS_CONFIG: Dict[str, str] = {
    "det.onnx": f"{HF_BASE_URL}/detection/v3/det.onnx",
    "rec_latin.onnx": f"{HF_BASE_URL}/languages/latin/rec.onnx",
    "dict_latin.txt": f"{HF_BASE_URL}/languages/latin/dict.txt",
    "rec_greek.onnx": f"{HF_BASE_URL}/languages/greek/rec.onnx",
    "dict_greek.txt": f"{HF_BASE_URL}/languages/greek/dict.txt",
}


def download_file(url: str, dest_path: Path) -> None:
    """Downloads a file with streaming if it does not already exist."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"File already exists: {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return

    logger.info(f"Downloading {dest_path.name} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        downloaded = 0
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        temp_path.rename(dest_path)
        logger.info(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size} bytes)")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Failed to download {dest_path.name}: {e}")
        raise


def ensure_models_downloaded(models_dir: Path) -> Dict[str, Path]:
    """Ensures all required ONNX models and dictionaries exist in the models directory."""
    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = {}

    for filename, url in MODELS_CONFIG.items():
        dest = models_dir / filename
        download_file(url, dest)
        downloaded_paths[filename] = dest

    return downloaded_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    target_models_dir = base_dir / "models"
    ensure_models_downloaded(target_models_dir)
    print("All models verified successfully!")
