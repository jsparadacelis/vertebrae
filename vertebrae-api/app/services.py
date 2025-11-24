import logging
from typing import Dict, Any, Optional

import numpy as np

from app.models import get_model, ModelType, get_all_models_info
from app.config import get_settings
from app.utils import load_image_from_bytes, draw_predictions_on_image, image_to_bytes

logger = logging.getLogger(__name__)
settings = get_settings()


def get_available_models() -> Dict[str, Any]:
    """
    Get information about all available models.

    Returns:
        Dictionary with models info and default model.
    """
    models_info = get_all_models_info()
    return {
        "models": models_info,
        "default_model": settings.default_model
    }


def get_specific_model_info(model_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about a specific model.

    Args:
        model_type: Type of model (yolo or maskrcnn). None uses default.

    Returns:
        Model information dictionary.

    Raises:
        ValueError: If model type is invalid.
    """
    model_type_enum = ModelType(model_type) if model_type else None
    model = get_model(model_type_enum)
    return model.get_model_info()


def predict_from_image_bytes(
    image_bytes: bytes,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run segmentation prediction on image bytes.

    Args:
        image_bytes: Raw image bytes.
        model_type: Type of model to use (yolo or maskrcnn). None uses default.

    Returns:
        Dictionary containing:
            - detections: List of formatted detection objects
            - num_detections: Number of vertebrae detected
            - image_shape: Original image dimensions
            - processing_time_ms: Inference time
            - model_used: Model type that was used

    Raises:
        ValueError: If image is invalid or model type is invalid.
        RuntimeError: If inference fails.
    """
    image = load_image_from_bytes(image_bytes)

    model_type_enum = ModelType(model_type) if model_type else None
    model = get_model(model_type_enum)

    predictions = model.predict(image)

    detections = model.format_detections(predictions)

    # Return structured result
    return {
        "detections": detections,
        "num_detections": predictions["num_detections"],
        "image_shape": predictions["image_shape"],
        "processing_time_ms": predictions["processing_time_ms"],
        "model_used": predictions.get(
            "model_type",
            model_type_enum.value if model_type_enum else settings.default_model
        )
    }


def predict_and_visualize(
    image_bytes: bytes,
    model_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run segmentation and generate annotated image.

    Args:
        image_bytes: Raw image bytes.
        model_type: Type of model to use (yolo or maskrcnn). None uses default.

    Returns:
        Dictionary containing:
            - image_bytes: Annotated image as PNG bytes
            - num_detections: Number of vertebrae detected
            - processing_time_ms: Inference time
            - model_used: Model type that was used

    Raises:
        ValueError: If image is invalid or model type is invalid.
        RuntimeError: If inference or visualization fails.
    """
    image = load_image_from_bytes(image_bytes)

    model_type_enum = ModelType(model_type) if model_type else None
    model = get_model(model_type_enum)

    predictions = model.predict(image)

    # Draw predictions on image
    annotated_image = draw_predictions_on_image(
        image=image,
        boxes=predictions["boxes"],
        masks=predictions["masks"],
        scores=predictions["scores"],
        classes=[settings.vertebrae_classes[i] for i in predictions["classes"]],
        score_threshold=settings.confidence_threshold
    )

    # Convert to bytes
    output_bytes = image_to_bytes(annotated_image, format="PNG")

    model_used = predictions.get(
        "model_type",
        model_type_enum.value if model_type_enum else settings.default_model
    )

    return {
        "image_bytes": output_bytes,
        "num_detections": predictions["num_detections"],
        "processing_time_ms": predictions["processing_time_ms"],
        "model_used": model_used
    }


def check_health() -> Dict[str, Any]:
    """
    Check health status of the service and models.

    Returns:
        Dictionary with health status information.
    """
    try:
        models_info = get_all_models_info()
        all_loaded = len(models_info) > 0

        return {
            "status": "healthy" if all_loaded else "unhealthy",
            "model_loaded": all_loaded,
            "model_path": str(settings.model_cache_dir) if all_loaded else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "model_path": None
        }
