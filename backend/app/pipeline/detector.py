"""
DBNet ONNX Text Detector.
Performs fast CPU-level text boundary detection using ONNX Runtime.
Postprocesses probability bitmaps, filters noise, and expands polygons via ClipperLib.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
import onnxruntime as ort

from backend.app.pipeline.preprocessor import ImagePreprocessor
from backend.app.utils.geometry import order_points_clockwise, unclip_polygon, sort_boxes_reading_order
from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.detector")


@dataclass
class DetectionBox:
    points: List[List[float]]  # 4 corner coordinates [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    score: float


class TextDetector:
    """ONNX Runtime-based DBNet text detector optimized for CPU inference."""

    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
        min_box_size: int = 3,
    ):
        self.model_path = model_path
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.min_box_size = min_box_size

        # Configure ONNX Runtime for CPU performance
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 2
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info(f"Loading detection ONNX model from {model_path} with {num_threads} CPU threads")
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(f"Detection model initialized successfully. Input: {self.input_name}")

    def _get_contour_score(self, prob_map: np.ndarray, contour: np.ndarray) -> float:
        """Calculates average probability value inside a contour."""
        h, w = prob_map.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
        score = cv2.mean(prob_map, mask=mask)[0]
        return float(score)

    def detect(self, image: np.ndarray, preprocessor: ImagePreprocessor) -> List[DetectionBox]:
        """
        Runs text detection on the input image:
        1. Preprocesses image to tensor
        2. Executes ONNX forward pass
        3. Thresholds bitmap and extracts contours
        4. Unclips polygons and scales coordinates to original image size
        5. Returns list of DetectionBox sorted in reading order
        """
        img_h, img_w = image.shape[:2]
        tensor, ratio_h, ratio_w = preprocessor.preprocess_for_detection(image)

        # Forward pass on CPU
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        prob_map = outputs[0][0, 0, :, :]

        # Threshold probability map
        bitmap = (prob_map > self.thresh).astype(np.uint8)

        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[np.ndarray] = []
        scores: List[float] = []

        for contour in contours:
            if len(contour) < 4:
                continue

            rect = cv2.minAreaRect(contour)
            w_box, h_box = rect[1]
            if min(w_box, h_box) < self.min_box_size:
                continue

            score = self._get_contour_score(prob_map, contour)
            if score < self.box_thresh:
                continue

            box = cv2.boxPoints(rect)
            box = order_points_clockwise(box)

            # Expand polygon outwards
            unclipped_box = unclip_polygon(box, unclip_ratio=self.unclip_ratio)
            if unclipped_box is None:
                continue

            # Scale back to original image dimensions
            unclipped_box[:, 0] *= ratio_w
            unclipped_box[:, 1] *= ratio_h

            # Clip within image boundaries
            unclipped_box[:, 0] = np.clip(unclipped_box[:, 0], 0, img_w - 1)
            unclipped_box[:, 1] = np.clip(unclipped_box[:, 1], 0, img_h - 1)

            boxes.append(unclipped_box)
            scores.append(score)

        if not boxes:
            logger.info("No text regions detected.")
            return []

        # Sort in natural reading order
        sorted_boxes = sort_boxes_reading_order(boxes)

        # Map back with their corresponding scores
        detection_results: List[DetectionBox] = []
        for s_box in sorted_boxes:
            # Find matching score
            matched_score = 0.5
            for idx, orig_b in enumerate(boxes):
                if np.allclose(s_box, orig_b, atol=1.0):
                    matched_score = scores[idx]
                    break

            detection_results.append(
                DetectionBox(
                    points=s_box.tolist(),
                    score=round(matched_score, 4),
                )
            )

        logger.info(f"Detected {len(detection_results)} text bounding boxes")
        return detection_results
