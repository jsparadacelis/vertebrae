from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):

    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")

    @field_validator('x1', 'y1', 'x2', 'y2')
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Coordinates must be non-negative")
        return v


class Detection(BaseModel):
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    mask: List[List[int]] = Field(..., description="Binary segmentation mask as 2D array")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    class_name: str = Field(..., description="Vertebra class (e.g., T1, L5)")
    class_id: int = Field(..., ge=0, description="Class index")


class PredictionResponse(BaseModel):
    detections: List[Detection] = Field(
        ...,
        description="List of detected vertebrae with masks"
    )
    num_detections: int = Field(..., description="Total number of detections")
    image_shape: List[int] = Field(
        ...,
        description="Input image dimensions [height, width, channels]"
    )
    processing_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    model_used: str = Field(..., description="Model used for inference (yolo or maskrcnn)")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: Optional[str] = Field(None, description="Path to cached model")


class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Model architecture name")
    model_type: str = Field(..., description="Model type (yolo or maskrcnn)")
    num_classes: int = Field(..., description="Number of vertebrae classes")
    classes: List[str] = Field(..., description="List of class names")
    backbone: str = Field(..., description="Model backbone architecture")
    device: str = Field(..., description="Inference device (cpu/cuda)")
    confidence_threshold: float = Field(..., description="Minimum confidence threshold")
    nms_threshold: float = Field(..., description="Non-maximum suppression threshold")
    framework: str = Field(..., description="Framework used (Ultralytics or Detectron2)")


class ModelsInfoResponse(BaseModel):
    models: Dict[str, Dict[str, Any]] = Field(..., description="Information about all loaded models")
    default_model: str = Field(..., description="Default model used when none specified")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Type of error")
