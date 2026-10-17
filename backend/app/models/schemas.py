"""
Pydantic schemas for API requests, responses, bounding boxes, and performance profiling.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class BoundingBoxPoint(BaseModel):
    x: float
    y: float


class RawDetection(BaseModel):
    box: List[List[float]] = Field(..., description="4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    text: str
    confidence: float


class FieldValue(BaseModel):
    value: Optional[str] = None
    confidence: float = 0.0
    source: str = Field(default="VIZ", description="Source of extraction: VIZ (visual zone) or MRZ")
    raw_match: Optional[str] = None


class ExtractedFields(BaseModel):
    full_name: FieldValue = Field(default_factory=FieldValue)
    document_number: FieldValue = Field(default_factory=FieldValue)
    date_of_birth: FieldValue = Field(default_factory=FieldValue)
    expiry_date: FieldValue = Field(default_factory=FieldValue)
    nationality: FieldValue = Field(default_factory=FieldValue)
    sex: FieldValue = Field(default_factory=FieldValue)
    personal_number: Optional[FieldValue] = None  # CNP for Romania, etc.
    document_type_detected: str = "Unknown"


class LatencyBreakdown(BaseModel):
    preprocessing_ms: float = 0.0
    detection_ms: float = 0.0
    recognition_ms: float = 0.0
    parsing_ms: float = 0.0
    total_ms: float = 0.0


class ExtractionResponse(BaseModel):
    success: bool
    document_type: str
    language_model_used: str
    fields: ExtractedFields
    raw_detections: List[RawDetection] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    image_metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class SampleImageInfo(BaseModel):
    filename: str
    locale: str
    dimensions: str
    url: str
