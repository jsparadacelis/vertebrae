"""YOLOv8 segmentation model implementation using Ultralytics."""

import logging
import time
from typing import Dict, List, Any

import numpy as np
from ultralytics import YOLO

from app.config import get_settings
from app.utils import download_model_from_s3
from app.models.base import BaseSegmentationModel

logger = logging.getLogger(__name__)
settings = get_settings()


class YOLOModel(BaseSegmentationModel):
    """YOLOv8 segmentation implementation for vertebrae segmentation."""

    def __init__(self):
        """Initialize the YOLO model."""
        super().__init__()
        self._model = None

    def load_model(self) -> None:
        """Load the YOLOv8 segmentation model from S3."""
        logger.info("Loading YOLOv8 segmentation model...")

        try:
            # Download model weights from S3
            model_path = download_model_from_s3(
                model_key=settings.yolo_model_key,
                cache_filename="yolo_model.pt"
            )
            logger.info(f"YOLOv8 weights loaded from {model_path}")

            # Load YOLO model
            self._model = YOLO(str(model_path))

            # Set device
            if settings.device == "cuda" and not self._model.device.type == "cuda":
                logger.warning("CUDA requested but not available, using CPU")

            self._model_loaded = True
            logger.info("YOLOv8 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise RuntimeError(f"YOLOv8 initialization failed: {e}")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an image using YOLOv8.

        Args:
            image: Input image as numpy array in BGR format.

        Returns:
            Dictionary containing predictions with bounding boxes, masks, scores, and classes.

        Raises:
            RuntimeError: If model is not loaded or inference fails.
        """
        if not self._model_loaded or self._model is None:
            raise RuntimeError("YOLOv8 model not loaded. Call load_model() first.")

        try:
            start_time = time.time()

            # Run inference
            # YOLOv8 expects BGR images (same as OpenCV)
            results = self._model.predict(
                image,
                conf=settings.confidence_threshold,
                iou=settings.nms_threshold,
                max_det=settings.max_detections,
                verbose=False
            )[0]  # Get first result (single image)

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Extract predictions
            boxes = []
            scores = []
            classes = []
            masks = []

            if results.boxes is not None and len(results.boxes) > 0:
                # Bounding boxes in xyxy format
                boxes = results.boxes.xyxy.cpu().numpy().tolist()
                # Confidence scores
                scores = results.boxes.conf.cpu().numpy().tolist()
                # Class IDs
                classes = results.boxes.cls.cpu().numpy().astype(int).tolist()

                # Segmentation masks
                if results.masks is not None:
                    # masks.data contains binary masks for each detection
                    masks_data = results.masks.data.cpu().numpy()  # Shape: (N, H, W)
                    masks = masks_data.astype(np.uint8)
                else:
                    # No masks available (shouldn't happen with seg model)
                    logger.warning("No masks in YOLOv8 prediction results")
                    masks = np.array([])

            num_detections = len(boxes)

            # Format predictions
            predictions = {
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
                "masks": masks,
                "num_detections": num_detections,
                "processing_time_ms": processing_time,
                "image_shape": list(image.shape),
                "model_type": "yolo"
            }

            logger.info(f"YOLOv8 inference completed in {processing_time:.2f}ms, found {num_detections} vertebrae")

            return predictions

        except Exception as e:
            logger.error(f"YOLOv8 inference failed: {e}")
            raise RuntimeError(f"YOLOv8 prediction failed: {e}")

    def format_detections(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format raw YOLOv8 predictions into structured detection objects.

        Args:
            predictions: Raw predictions from model.

        Returns:
            List of formatted detection dictionaries.
        """
        detections = []

        boxes = predictions["boxes"]
        scores = predictions["scores"]
        classes = predictions["classes"]
        masks = predictions["masks"]

        for i in range(predictions["num_detections"]):
            box = boxes[i]
            score = scores[i]
            class_id = classes[i]
            mask = masks[i]

            # Get class name
            class_name = settings.vertebrae_classes[class_id]

            detection = {
                "bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3])
                },
                "mask": mask.tolist(),
                "score": float(score),
                "class_name": class_name,
                "class_id": int(class_id)
            }

            detections.append(detection)

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the YOLOv8 model."""
        model_variant = "YOLOv8m-seg" if "m" in settings.yolo_model_key else "YOLOv8-seg"

        return {
            "model_name": model_variant,
            "model_type": "yolo",
            "num_classes": settings.num_classes,
            "classes": settings.vertebrae_classes,
            "backbone": "YOLOv8",
            "device": settings.device,
            "confidence_threshold": settings.confidence_threshold,
            "nms_threshold": settings.nms_threshold,
            "framework": "Ultralytics"
        }
