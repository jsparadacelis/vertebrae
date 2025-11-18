"""FastAPI application for vertebrae segmentation with multiple model support."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Query
from fastapi.responses import Response

from app.config import get_settings
from app.models import initialize_all_models
from app.services import SegmentationService
from app.schemas import (
    PredictionResponse,
    Detection,
    HealthResponse,
    ModelInfo,
    ErrorResponse,
    ModelsInfoResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()
segmentation_service = SegmentationService()


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
    health_status = segmentation_service.check_health()
    return HealthResponse(**health_status)


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
        models_info = segmentation_service.get_available_models()
        return ModelsInfoResponse(**models_info)
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
        info = segmentation_service.get_specific_model_info(model)
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
        # Read image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )

        # Run prediction
        result = segmentation_service.predict_from_image_bytes(image_bytes, model)

        # Create response
        response = PredictionResponse(
            detections=[Detection(**det) for det in result["detections"]],
            num_detections=result["num_detections"],
            image_shape=result["image_shape"],
            processing_time_ms=result["processing_time_ms"],
            model_used=result["model_used"]
        )

        logger.info(f"Prediction successful using {result['model_used']}: {result['num_detections']} vertebrae detected")
        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
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
        # Read image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )

        # Run prediction and visualization
        result = segmentation_service.predict_and_visualize(image_bytes, model)

        logger.info(f"Visualization successful using {result['model_used']}: {result['num_detections']} vertebrae annotated")

        return Response(
            content=result["image_bytes"],
            media_type="image/png",
            headers={
                "X-Num-Detections": str(result["num_detections"]),
                "X-Processing-Time-Ms": str(result["processing_time_ms"]),
                "X-Model-Used": result["model_used"]
            }
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
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
