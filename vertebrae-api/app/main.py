"""FastAPI application for vertebrae segmentation with multiple model support."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query
from fastapi.responses import Response

from app.config import get_settings
from models.base import BaseSegmentationModel
from models.factory import get_model, ModelType
from app.schemas import (
    PredictionResponse,
    Detection,
    HealthResponse,
    ModelInfo,
    ErrorResponse,
    ModelsInfoResponse
)
from app.utils import (
    load_image_from_bytes,
    draw_predictions_on_image,
    image_to_bytes
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application."""
    # Startup: Load all models
    logger.info("Starting up vertebrae segmentation API...")
    try:
        initialize_all_models()
        logger.info("All models loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down vertebrae segmentation API...")


# Create FastAPI app
app = FastAPI(
    title="Vertebrae Segmentation API",
    description="Multi-model API for vertebrae segmentation (T1-T12, L1-L5) supporting YOLO and Mask R-CNN",
    version="0.2.0",
    lifespan=lifespan
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Vertebrae Segmentation API",
        "version": "0.2.0",
        "description": "Multi-model segmentation for 17 vertebrae classes",
        "supported_models": ["yolo", "maskrcnn"],
        "default_model": settings.default_model,
        "endpoints": {
            "POST /predict": "Run segmentation inference (supports ?model=yolo or ?model=maskrcnn)",
            "POST /predict/visualize": "Run inference and return annotated image",
            "GET /health": "Health check",
            "GET /models": "Information about all available models",
            "GET /model-info": "Information about specific model"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check if the API and models are healthy.

    Returns:
        Health status with model loading state.
    """
    try:
        models_info = get_all_models_info()
        all_loaded = len(models_info) > 0

        return HealthResponse(
            status="healthy" if all_loaded else "unhealthy",
            model_loaded=all_loaded,
            model_path=str(settings.model_cache_dir) if all_loaded else None
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_path=None
        )


@app.get(
    "/models",
    response_model=ModelsInfoResponse,
    tags=["Model"],
    summary="Get information about all available models"
)
async def get_models_info():
    """
    Get information about all loaded models.

    Returns:
        Information about all available models.
    """
    try:
        models_info = get_all_models_info()
        return ModelsInfoResponse(
            models=models_info,
            default_model=settings.default_model
        )
    except Exception as e:
        logger.error(f"Failed to get models info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models information: {str(e)}"
        )


@app.get(
    "/model-info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Get information about a specific model"
)
async def get_model_info(
    model: Optional[str] = Query(None, description="Model type (yolo or maskrcnn)")
):
    """
    Get information about a specific model.

    Args:
        model: Model type to query (defaults to configured default).

    Returns:
        Model configuration and metadata.
    """
    try:
        model_type = ModelType(model) if model else None
        model_instance = get_model(model_type)
        info = model_instance.get_model_info()
        return ModelInfo(**info)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model type: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Run vertebrae segmentation",
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid image or model type"},
        500: {"model": ErrorResponse, "description": "Inference failed"}
    }
)
async def predict(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    model: Optional[str] = Query(None, description="Model to use (yolo or maskrcnn, defaults to configured default)")
):
    """
    Run vertebrae segmentation on an uploaded image.

    Args:
        file: Uploaded image file.
        model: Model type to use for inference.

    Returns:
        Predictions with bounding boxes, masks (RLE), scores, and class labels.

    Raises:
        HTTPException: If image is invalid or inference fails.
    """
    try:
        # Read and validate image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )

        # Load image
        try:
            image = load_image_from_bytes(image_bytes)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {str(e)}"
            )

        # Get model
        try:
            model_type = ModelType(model) if model else None
            model_instance = get_model(model_type)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type '{model}'. Use 'yolo' or 'maskrcnn'"
            )

        # Run inference
        predictions = model_instance.predict(image)

        # Format detections
        detections = model_instance.format_detections(predictions)

        # Create response
        response = PredictionResponse(
            detections=[Detection(**det) for det in detections],
            num_detections=predictions["num_detections"],
            image_shape=predictions["image_shape"],
            processing_time_ms=predictions["processing_time_ms"],
            model_used=predictions.get("model_type", model_type.value if model_type else settings.default_model)
        )

        logger.info(f"Prediction successful using {predictions.get('model_type', 'unknown')}: {predictions['num_detections']} vertebrae detected")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post(
    "/predict/visualize",
    tags=["Inference"],
    summary="Run segmentation and return annotated image",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Annotated image with predictions"
        },
        400: {"model": ErrorResponse, "description": "Invalid image or model type"},
        500: {"model": ErrorResponse, "description": "Inference failed"}
    }
)
async def predict_visualize(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    model: Optional[str] = Query(None, description="Model to use (yolo or maskrcnn, defaults to configured default)")
):
    """
    Run vertebrae segmentation and return an annotated image.

    Args:
        file: Uploaded image file.
        model: Model type to use for inference.

    Returns:
        PNG image with bounding boxes, masks, and labels drawn.

    Raises:
        HTTPException: If image is invalid or inference fails.
    """
    try:
        # Read and validate image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )

        # Load image
        try:
            image = load_image_from_bytes(image_bytes)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format: {str(e)}"
            )

        # Get model
        try:
            model_type = ModelType(model) if model else None
            model_instance = get_model(model_type)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type '{model}'. Use 'yolo' or 'maskrcnn'"
            )

        # Run inference
        predictions = model_instance.predict(image)

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

        model_used = predictions.get("model_type", model_type.value if model_type else settings.default_model)
        logger.info(f"Visualization successful using {model_used}: {predictions['num_detections']} vertebrae annotated")

        return Response(
            content=output_bytes,
            media_type="image/png",
            headers={
                "X-Num-Detections": str(predictions["num_detections"]),
                "X-Processing-Time-Ms": str(predictions["processing_time_ms"]),
                "X-Model-Used": model_used
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visualization failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower()
    )
