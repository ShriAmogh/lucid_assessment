"""
Image Preprocessor module for DBNet Detection and SVTR Recognition.
Handles RGBA alpha flattening, aspect-ratio preserving resizing with 32-px alignment,
4-point perspective crop rectification, and tensor normalization.
"""

import io
import math
from typing import Tuple, List, Union
import numpy as np
import cv2
from PIL import Image

from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.preprocessor")


class ImagePreprocessor:
    """Preprocesses images for ONNX detection and recognition pipelines."""

    def __init__(self, det_limit_side_len: int = 960):
        self.det_limit_side_len = det_limit_side_len
        # Standard ImageNet / PaddleOCR normalization factors
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Decodes image bytes into a 3-channel RGB numpy array.
        Handles RGBA transparency by alpha compositing over a pure white background.
        """
        pil_img = Image.open(io.BytesIO(image_bytes))

        if pil_img.mode == "RGBA":
            # Composite over white background
            background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(background, pil_img)
            img = np.array(alpha_composite.convert("RGB"))
        elif pil_img.mode != "RGB":
            img = np.array(pil_img.convert("RGB"))
        else:
            img = np.array(pil_img)

        logger.debug(f"Loaded image with shape {img.shape}")
        return img

    def preprocess_for_detection(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Prepares image for DBNet ONNX detection:
        1. Resizes such that the maximum edge is <= det_limit_side_len, and both H, W are multiples of 32.
        2. Normalizes to float32 tensor of shape (1, 3, H, W).
        Returns:
            tensor: (1, 3, H, W)
            ratio_h: original_height / resized_height
            ratio_w: original_width / resized_width
        """
        h, w = image.shape[:2]
        ratio = 1.0
        max_edge = max(h, w)

        if max_edge > self.det_limit_side_len:
            ratio = float(self.det_limit_side_len) / float(max_edge)

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # DBNet requires H and W to be divisible by 32
        resize_h = max(int(round(resize_h / 32.0) * 32), 32)
        resize_w = max(int(round(resize_w / 32.0) * 32), 32)

        resized_img = cv2.resize(image, (resize_w, resize_h))

        # Scale factors to map detected boxes back to original image coordinates
        ratio_h = float(h) / float(resize_h)
        ratio_w = float(w) / float(resize_w)

        # Normalize: (X / 255.0 - mean) / std
        norm_img = resized_img.astype(np.float32) / 255.0
        norm_img = (norm_img - self.mean) / self.std

        # Transpose HWC -> CHW and add batch dimension -> NCHW
        tensor = norm_img.transpose((2, 0, 1))[np.newaxis, :]
        return tensor.astype(np.float32), ratio_h, ratio_w

    def crop_box(self, image: np.ndarray, points: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Extracts an oriented bounding box from an image using 4-point perspective warp.
        Points must be ordered: [top-left, top-right, bottom-right, bottom-left].
        """
        pts = np.array(points, dtype=np.float32)

        # Calculate bounding width and height
        width_top = np.linalg.norm(pts[0] - pts[1])
        width_bottom = np.linalg.norm(pts[3] - pts[2])
        crop_w = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(pts[0] - pts[3])
        height_right = np.linalg.norm(pts[1] - pts[2])
        crop_h = int(max(height_left, height_right))

        crop_w = max(crop_w, 4)
        crop_h = max(crop_h, 4)

        # Destination rectangle
        dst_pts = np.array([
            [0, 0],
            [crop_w - 1, 0],
            [crop_w - 1, crop_h - 1],
            [0, crop_h - 1],
        ], dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(pts, dst_pts)
        crop = cv2.warpPerspective(
            image,
            transform_matrix,
            (crop_w, crop_h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )

        # If vertical text (h > 1.5 * w), rotate 90 degrees clockwise
        if crop_h > crop_w * 1.5:
            crop = np.rot90(crop, -1)

        return crop

    def preprocess_for_recognition(self, crop: np.ndarray, target_height: int = 48) -> np.ndarray:
        """
        Prepares a cropped text patch for SVTR recognition:
        1. Fixed height = target_height (48)
        2. Proportional dynamic width
        3. Normalized to [-0.5, 0.5] range (standard PaddleOCR SVTR normalization)
        Returns:
            tensor: (1, 3, target_height, target_width)
        """
        h, w = crop.shape[:2]
        ratio = float(w) / float(h)
        target_width = max(int(math.ceil(target_height * ratio)), 8)

        resized_crop = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # Normalize: (X / 255.0 - 0.5) / 0.5 -> [-1.0, 1.0] or [0, 1] - 0.5
        norm_crop = resized_crop.astype(np.float32) / 255.0
        norm_crop = (norm_crop - 0.5) / 0.5

        # Transpose HWC -> CHW -> NCHW
        tensor = norm_crop.transpose((2, 0, 1))[np.newaxis, :]
        return tensor.astype(np.float32)
