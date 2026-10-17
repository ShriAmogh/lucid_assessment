"""
High-resolution Pipeline Benchmarker.
Measures and logs sub-millisecond CPU latency across each stage of the OCR pipeline.
"""

import time
from contextlib import contextmanager
from typing import Dict
from backend.app.models.schemas import LatencyBreakdown
from backend.app.utils.logger import get_logger

logger = get_logger("lucid_ocr.benchmarker")


class PipelineBenchmarker:
    """Tracks latency for each phase of the OCR pipeline."""

    def __init__(self):
        self.timings: Dict[str, float] = {
            "preprocessing": 0.0,
            "detection": 0.0,
            "recognition": 0.0,
            "parsing": 0.0,
            "total": 0.0,
        }
        self._total_start: float = 0.0

    def start_total(self) -> None:
        """Starts overall timer for the entire pipeline."""
        self._total_start = time.perf_counter()

    def stop_total(self) -> float:
        """Stops overall timer and calculates total duration in milliseconds."""
        if self._total_start > 0:
            elapsed_ms = (time.perf_counter() - self._total_start) * 1000.0
            self.timings["total"] = round(elapsed_ms, 2)
        return self.timings["total"]

    @contextmanager
    def measure(self, phase_name: str):
        """Context manager to measure execution time of a specific phase."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.timings[phase_name] = round(elapsed_ms, 2)
            logger.debug(f"Phase '{phase_name}' completed in {elapsed_ms:.2f} ms")

    def get_breakdown(self) -> LatencyBreakdown:
        """Returns structured LatencyBreakdown model."""
        return LatencyBreakdown(
            preprocessing_ms=self.timings.get("preprocessing", 0.0),
            detection_ms=self.timings.get("detection", 0.0),
            recognition_ms=self.timings.get("recognition", 0.0),
            parsing_ms=self.timings.get("parsing", 0.0),
            total_ms=self.timings.get("total", 0.0),
        )
