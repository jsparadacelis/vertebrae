"""FastAPI application for vertebrae segmentation."""

import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import Response, JSONResponse

from app.config import get_settings
from app.model import get_model
from app.schemas import (
    PredictionResponse,
    Detection,
    HealthResponse,
    ModelInfo,
    ErrorResponse
)
from app.utils import (
    load_image_from_bytes,
    draw_predictions_on_image,
    image_to_bytes,
    decode_rle_to_mask
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
    # Startup: Load model
    logger.info("Starting up vertebrae segmentation API...")
    try:
        model = get_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down vertebrae segmentation API...")


# Create FastAPI app
app = FastAPI(
    title="Vertebrae Segmentation API",
    description="Mask R-CNN based API for vertebrae segmentation (T1-T12, L1-L5)",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Vertebrae Segmentation API",
        "version": "0.1.0",
        "description": "Mask R-CNN based segmentation for 17 vertebrae classes",
        "endpoints": {
            "POST /predict": "Run segmentation inference",
            "POST /predict/visualize": "Run inference and return annotated image",
            "GET /health": "Health check",
            "GET /model-info": "Model information"
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
    Check if the API and model are healthy.

    Returns:
        Health status with model loading state.
    """
    try:
        model = get_model()
        is_loaded = model.is_loaded()

        return HealthResponse(
            status="healthy" if is_loaded else "unhealthy",
            model_loaded=is_loaded,
            model_path=str(settings.get_model_path()) if is_loaded else None
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_path=None
        )


@app.get(
    "/model-info",
    response_model=ModelInfo,
    tags=["Model"],
    summary="Get model information"
)
async def get_model_info():
    """
    Get information about the loaded model.

    Returns:
        Model configuration and metadata.
    """
    try:
        model = get_model()
        info = model.get_model_info()
        return ModelInfo(**info)
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
        400: {"model": ErrorResponse, "description": "Invalid image format"},
        500: {"model": ErrorResponse, "description": "Inference failed"}
    }
)
async def predict(file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)")):
    """
    Run vertebrae segmentation on an uploaded image.

    Args:
        file: Uploaded image file.

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

        # Run inference
        model = get_model()
        predictions = model.predict(image)

        # Format detections
        detections = model.format_detections(predictions)

        # Create response
        response = PredictionResponse(
            detections=[Detection(**det) for det in detections],
            num_detections=predictions["num_detections"],
            image_shape=predictions["image_shape"],
            processing_time_ms=predictions["processing_time_ms"]
        )

        logger.info(f"Prediction successful: {predictions['num_detections']} vertebrae detected")
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
        400: {"model": ErrorResponse, "description": "Invalid image format"},
        500: {"model": ErrorResponse, "description": "Inference failed"}
    }
)
async def predict_visualize(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)")
):
    """
    Run vertebrae segmentation and return an annotated image.

    Args:
        file: Uploaded image file.

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

        # Run inference
        model = get_model()
        predictions = model.predict(image)

        # Decode RLE masks back to binary masks for visualization
        masks_binary = []
        for i in range(predictions["num_detections"]):
            mask_rle = predictions["masks"][i]
            # Convert numpy mask to RLE first
            from app.utils import encode_mask_to_rle
            rle = encode_mask_to_rle(mask_rle)
            masks_binary.append(mask_rle)

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

        logger.info(f"Visualization successful: {predictions['num_detections']} vertebrae annotated")

        return Response(
            content=output_bytes,
            media_type="image/png",
            headers={
                "X-Num-Detections": str(predictions["num_detections"]),
                "X-Processing-Time-Ms": str(predictions["processing_time_ms"])
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
