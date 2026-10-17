"""
Complete End-to-End OCR & Key-Value Extraction Engine.
Coordinates Preprocessing, DBNet Detection, SVTR Multilingual Recognition,
Key-Value Parsing, and Latency Profiling.
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from backend.app.config import settings
from backend.app.pipeline.preprocessor import ImagePreprocessor
from backend.app.pipeline.detector import TextDetector, DetectionBox
from backend.app.pipeline.recognizer import TextRecognizer, RecognizedText
from backend.app.pipeline.kv_extractor import KeyValueExtractor
from backend.app.pipeline.benchmarker import PipelineBenchmarker
from backend.app.models.schemas import (
    ExtractionResponse,
    RawDetection,
    ExtractedFields,
    LatencyBreakdown,
)
from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.engine")


class OCREngine:
    """Orchestrates the entire offline ONNX CPU OCR pipeline."""

    def __init__(self):
        logger.info("Initializing LucidPlus OCR Engine...")
        self.preprocessor = ImagePreprocessor(
            det_limit_side_len=settings.models.det_limit_side_len
        )

        logger.info("Loading ONNX Detection model...")
        self.detector = TextDetector(
            model_path=str(settings.models.det_model_path),
            num_threads=settings.models.intra_op_num_threads,
            thresh=settings.models.det_db_thresh,
            box_thresh=settings.models.det_db_box_thresh,
            unclip_ratio=settings.models.det_db_unclip_ratio,
            min_box_size=settings.models.min_box_size,
        )

        logger.info("Loading ONNX Latin Recognition model (English, German, Romanian)...")
        self.recognizer_latin = TextRecognizer(
            model_path=str(settings.models.rec_latin_model_path),
            dict_path=str(settings.models.dict_latin_path),
            num_threads=settings.models.intra_op_num_threads,
        )

        logger.info("Loading ONNX Greek Recognition model...")
        self.recognizer_greek = TextRecognizer(
            model_path=str(settings.models.rec_greek_model_path),
            dict_path=str(settings.models.dict_greek_path),
            num_threads=settings.models.intra_op_num_threads,
        )

        self.kv_extractor = KeyValueExtractor()
        logger.info("LucidPlus OCR Engine initialized successfully!")

    def select_recognizer(self, language_hint: str) -> Tuple[TextRecognizer, str]:
        """Selects the optimal recognition model based on language hint."""
        hint = language_hint.lower().strip()
        if "greek" in hint:
            return self.recognizer_greek, "PP-OCRv4 Greek SVTR (Greek + Latin)"
        else:
            return self.recognizer_latin, "PP-OCRv4 Latin SVTR (German, Romanian, English)"

    def process_image(
        self,
        image_bytes: bytes,
        language_hint: str = "auto",
    ) -> ExtractionResponse:
        """
        Executes end-to-end OCR and Key-Value extraction:
        1. Preprocesses input image.
        2. Detects text bounding boxes via DBNet ONNX.
        3. Crops text patches and runs SVTR multilingual recognition.
        4. Extracts semantic key-values (Full Name, Doc No, DOB, Expiry, Nationality, Sex).
        5. Profiles sub-millisecond CPU latency across all phases.
        """
        benchmarker = PipelineBenchmarker()
        benchmarker.start_total()

        try:
            # Phase 1: Preprocessing
            with benchmarker.measure("preprocessing"):
                img = self.preprocessor.load_from_bytes(image_bytes)
                img_h, img_w = img.shape[:2]

            # Phase 2: Text Detection
            with benchmarker.measure("detection"):
                detection_boxes = self.detector.detect(img, self.preprocessor)

            if not detection_boxes:
                benchmarker.stop_total()
                return ExtractionResponse(
                    success=True,
                    document_type="Unknown",
                    language_model_used="None",
                    fields=ExtractedFields(),
                    raw_detections=[],
                    latency=benchmarker.get_breakdown(),
                    image_metadata={"width": img_w, "height": img_h, "channels": 3},
                )

            # Phase 3: Perspective Crop & Recognition
            with benchmarker.measure("recognition"):
                crops = [
                    self.preprocessor.crop_box(img, box.points)
                    for box in detection_boxes
                ]

                # If auto-detect, determine if Greek or Latin
                rec_model, model_label = self.select_recognizer(language_hint)

                if language_hint.lower() == "auto":
                    # Run on first 2 header crops to check for Greek characters
                    header_crops = crops[:min(3, len(crops))]
                    prelim_texts = self.recognizer_greek.recognize_crops(
                        header_crops, self.preprocessor
                    )
                    has_greek_chars = any(
                        any("\u0370" <= c <= "\u03ff" or "\u1f00" <= c <= "\u1fff" for c in pt.text)
                        for pt in prelim_texts
                    )
                    if has_greek_chars:
                        rec_model = self.recognizer_greek
                        model_label = "PP-OCRv4 Greek SVTR (Auto-Routed)"
                    else:
                        rec_model = self.recognizer_latin
                        model_label = "PP-OCRv4 Latin SVTR (Auto-Routed)"

                recognized_texts = rec_model.recognize_crops(
                    crops, self.preprocessor, detection_boxes
                )

            # Phase 4: Key-Value Extraction
            with benchmarker.measure("parsing"):
                fields = self.kv_extractor.extract(recognized_texts)

            benchmarker.stop_total()

            # Format raw detections for UI overlay
            raw_dets = [
                RawDetection(
                    box=rt.box or [],
                    text=rt.text,
                    confidence=rt.confidence,
                )
                for rt in recognized_texts
            ]

            return ExtractionResponse(
                success=True,
                document_type=fields.document_type_detected,
                language_model_used=model_label,
                fields=fields,
                raw_detections=raw_dets,
                latency=benchmarker.get_breakdown(),
                image_metadata={"width": img_w, "height": img_h, "channels": 3},
            )

        except Exception as e:
            logger.error(f"Error during OCR processing: {e}", exc_info=True)
            benchmarker.stop_total()
            return ExtractionResponse(
                success=False,
                document_type="Error",
                language_model_used="Error",
                fields=ExtractedFields(),
                raw_detections=[],
                latency=benchmarker.get_breakdown(),
                image_metadata={},
                error=str(e),
            )
