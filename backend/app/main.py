"""
FastAPI Backend Application for LucidPlus Offline CPU OCR.
Exposes RESTful endpoints for image upload, inference, key-value extraction,
and preloaded sample inspection.
"""

import os
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from backend.app.config import settings, IMAGES_DIR
from backend.app.pipeline.ocr_engine import OCREngine
from backend.app.models.schemas import ExtractionResponse, SampleImageInfo
from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.api")

# Singleton OCR Engine instance
engine: Optional[OCREngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models on startup and cleans up on shutdown."""
    global engine
    logger.info("Initializing LucidPlus OCR Engine during application startup...")
    engine = OCREngine()
    yield
    logger.info("Shutting down LucidPlus OCR API...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Offline-capable, CPU-optimized ONNX OCR pipeline for international ID cards.",
    lifespan=lifespan,
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
def root():
    """Root endpoint providing service summary and documentation links."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "endpoints": {
            "extract": "POST /api/extract",
            "samples": "GET /api/samples",
            "health": "GET /api/health",
        },
    }


@app.get("/api/health", tags=["System"])
def health_check():
    """Health check and model status inspection."""
    global engine
    if engine is None:
        engine = OCREngine()

    return {
        "status": "ok",
        "cpu_threads": settings.models.intra_op_num_threads,
        "models_loaded": {
            "detection": engine.detector is not None,
            "latin_rec": engine.recognizer_latin is not None,
            "greek_rec": engine.recognizer_greek is not None,
        },
    }


@app.get("/api/samples", response_model=List[SampleImageInfo], tags=["Samples"])
def list_samples():
    """Returns metadata for all bundled international ID samples in the images directory."""
    if not IMAGES_DIR.exists():
        return []

    samples = []
    for file_path in sorted(IMAGES_DIR.glob("*.png")):
        filename = file_path.name

        # Determine locale from filename
        locale = "Unknown"
        if "german" in filename:
            locale = "German ID"
        elif "greek" in filename:
            locale = "Greek ID"
        elif "roman" in filename:
            locale = "Romanian ID"
        elif "english" in filename:
            locale = "English / UK ID"

        # Read dimensions
        try:
            with Image.open(file_path) as im:
                dims = f"{im.width}x{im.height}"
        except Exception:
            dims = "Unknown"

        samples.append(
            SampleImageInfo(
                filename=filename,
                locale=locale,
                dimensions=dims,
                url=f"/api/samples/{filename}",
            )
        )

    return samples


@app.get("/api/samples/{filename}", tags=["Samples"])
def get_sample_image(filename: str):
    """Serves the raw image file for a given sample."""
    file_path = IMAGES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample image '{filename}' not found.",
        )
    return FileResponse(file_path, media_type="image/png")


@app.post("/api/extract", response_model=ExtractionResponse, tags=["Inference"])
async def extract_key_values(
    file: UploadFile = File(..., description="Document image file (PNG, JPG, WebP)"),
    language_hint: str = Form(
        default="auto",
        description="Language hint: auto, german, greek, romanian, english",
    ),
):
    """
    Primary OCR endpoint:
    Processes uploaded identity document and extracts structured key-value pairs (Full Name,
    Document Number, Date of Birth, Expiry Date, Nationality, Sex) along with CPU latency breakdown.
    """
    global engine
    if engine is None:
        engine = OCREngine()

    logger.info(f"Received extraction request for file: {file.filename}, language_hint: {language_hint}")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded.",
            )

        response = engine.process_image(image_bytes, language_hint=language_hint)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error processing {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing OCR pipeline: {str(e)}",
        )
