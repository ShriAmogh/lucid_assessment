"""
SVTR-LCNet ONNX Text Recognizer.
Performs multilingual character recognition with CTC greedy decoding on cropped text regions.
Supports Greek, German, Romanian, and English alphabets with diacritics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import onnxruntime as ort

from backend.app.pipeline.preprocessor import ImagePreprocessor
from backend.app.pipeline.detector import DetectionBox
from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.recognizer")


@dataclass
class RecognizedText:
    text: str
    confidence: float
    box: Optional[List[List[float]]] = None


class TextRecognizer:
    """ONNX Runtime-based text recognition model with CTC greedy decoder."""

    def __init__(self, model_path: str, dict_path: str, num_threads: int = 4):
        self.model_path = model_path
        self.dict_path = dict_path
        self.characters = self._load_dict(dict_path)

        # Configure ONNX session for CPU execution
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = num_threads
        session_options.inter_op_num_threads = 2
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info(f"Loading recognition ONNX model: {model_path} with dict {dict_path}")
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Verify dictionary length matches output dimension
        out_dim = self.session.get_outputs()[0].shape[-1]
        logger.info(f"Model initialized: output vocab size = {out_dim}, loaded characters = {len(self.characters)}")

    def _load_dict(self, dict_path: str) -> List[str]:
        """
        Loads character dictionary for CTC decoding.
        Index 0 is the blank token ('').
        Dictionary characters follow line by line.
        Appends space (' ') and fallback padding to match output tensor dimension if needed.
        """
        chars = [""]  # Blank token at index 0

        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\r\n")
                chars.append(line)

        # PaddleOCR appends a space character at the end of the vocabulary
        chars.append(" ")
        return chars

    def decode_ctc(self, logits: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Performs CTC greedy argmax decoding on model predictions of shape (N, T, num_classes):
        1. Checks if outputs are already softmax probabilities or raw logits.
        2. Argmax over vocabulary dimension.
        3. Collapses consecutive duplicate characters.
        4. Strips blank tokens (index 0).
        """
        # If model output is already softmax probabilities (sums to ~1 along last axis)
        if np.all(logits >= 0.0) and np.isclose(logits[0, 0].sum(), 1.0, atol=1e-2):
            probs = logits
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        preds_idx = np.argmax(probs, axis=-1)   # (N, T)
        preds_prob = np.max(probs, axis=-1)    # (N, T)

        batch_size, time_steps = preds_idx.shape
        decoded_texts: List[str] = []
        decoded_confs: List[float] = []

        num_chars = len(self.characters)

        for b in range(batch_size):
            char_list: List[str] = []
            conf_list: List[float] = []
            prev_idx = -1

            for t in range(time_steps):
                idx = preds_idx[b, t]
                # CTC collapse rule: skip consecutive identical tokens or blank (0)
                if idx != prev_idx and idx != 0:
                    if idx < num_chars:
                        char_list.append(self.characters[idx])
                        conf_list.append(float(preds_prob[b, t]))
                prev_idx = idx

            text = "".join(char_list).strip()
            conf = float(np.mean(conf_list)) if conf_list else 0.0
            decoded_texts.append(text)
            decoded_confs.append(round(conf, 4))

        return decoded_texts, decoded_confs

    def recognize_crops(
        self,
        crops: List[np.ndarray],
        preprocessor: ImagePreprocessor,
        boxes: Optional[List[DetectionBox]] = None,
        batch_size: int = 16,
    ) -> List[RecognizedText]:
        """
        Performs batch recognition on a list of image crops:
        1. Preprocesses crops to shape (1, 3, 48, W)
        2. Batches crops with similar widths to minimize padding waste
        3. Runs ONNX inference
        4. Decodes text strings and binds with bounding boxes
        """
        if not crops:
            return []

        results: List[RecognizedText] = []

        # Process in batches
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            batch_boxes = boxes[i : i + batch_size] if boxes else [None] * len(batch_crops)

            # Find max width in this batch after fixed-height resize
            processed_tensors = [
                preprocessor.preprocess_for_recognition(crop, target_height=48)
                for crop in batch_crops
            ]

            max_w = max(t.shape[3] for t in processed_tensors)
            # Ensure width is at least 32
            max_w = max(max_w, 32)

            # Pad all batch items to max_w
            batch_tensor = np.zeros((len(batch_crops), 3, 48, max_w), dtype=np.float32)
            for b_idx, tensor in enumerate(processed_tensors):
                curr_w = tensor.shape[3]
                batch_tensor[b_idx, :, :, :curr_w] = tensor[0]

            # ONNX forward pass
            outputs = self.session.run([self.output_name], {self.input_name: batch_tensor})
            logits = outputs[0]  # (B, T, num_classes)

            texts, confs = self.decode_ctc(logits)

            for text, conf, box_obj in zip(texts, confs, batch_boxes):
                box_pts = box_obj.points if box_obj else None
                results.append(
                    RecognizedText(
                        text=text,
                        confidence=conf,
                        box=box_pts,
                    )
                )

        return results
