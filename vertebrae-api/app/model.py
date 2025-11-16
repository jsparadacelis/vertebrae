"""Detectron2 model loading and inference for vertebrae segmentation."""

import logging
import time
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from app.config import get_settings
from app.utils import download_model_from_s3, encode_mask_to_rle

logger = logging.getLogger(__name__)
settings = get_settings()


class VertebraeSegmentationModel:
    """Singleton class for Mask R-CNN vertebrae segmentation model."""

    _instance: Optional['VertebraeSegmentationModel'] = None
    _predictor: Optional[DefaultPredictor] = None
    _model_loaded: bool = False

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model (only once due to singleton)."""
        if not self._model_loaded:
            self.load_model()

    def load_model(self) -> None:
        """Load the Mask R-CNN model from S3 and configure Detectron2."""
        logger.info("Loading Mask R-CNN model...")

        try:
            # Download model weights from S3
            model_path = download_model_from_s3()
            logger.info(f"Model weights loaded from {model_path}")

            # Configure Detectron2
            cfg = get_cfg()

            # Load base configuration for Mask R-CNN R50-FPN
            cfg.merge_from_file(
                model_zoo.get_config_file(settings.model_backbone)
            )

            # Model configuration
            cfg.MODEL.WEIGHTS = str(model_path)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = settings.num_classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = settings.confidence_threshold
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = settings.nms_threshold
            cfg.MODEL.DEVICE = settings.device

            # Additional inference settings
            cfg.TEST.DETECTIONS_PER_IMAGE = settings.max_detections

            # Create predictor
            self._predictor = DefaultPredictor(cfg)
            self._model_loaded = True

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an image.

        Args:
            image: Input image as numpy array in BGR format.

        Returns:
            Dictionary containing predictions with bounding boxes, masks, scores, and classes.

        Raises:
            RuntimeError: If model is not loaded or inference fails.
        """
        if not self._model_loaded or self._predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            start_time = time.time()

            # Run inference
            outputs = self._predictor(image)

            # Extract predictions
            instances = outputs["instances"].to("cpu")

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            # Format predictions
            predictions = {
                "boxes": instances.pred_boxes.tensor.numpy().tolist() if len(instances) > 0 else [],
                "scores": instances.scores.numpy().tolist() if len(instances) > 0 else [],
                "classes": instances.pred_classes.numpy().tolist() if len(instances) > 0 else [],
                "masks": instances.pred_masks.numpy() if len(instances) > 0 else np.array([]),
                "num_detections": len(instances),
                "processing_time_ms": processing_time,
                "image_shape": list(image.shape)
            }

            logger.info(f"Inference completed in {processing_time:.2f}ms, found {len(instances)} vertebrae")

            return predictions

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def format_detections(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format raw predictions into structured detection objects.

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

            # Encode mask to RLE
            mask_rle = encode_mask_to_rle(mask)

            detection = {
                "bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3])
                },
                "mask": mask_rle,
                "score": float(score),
                "class_name": class_name,
                "class_id": int(class_id)
            }

            detections.append(detection)

        return detections

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": "Mask R-CNN",
            "num_classes": settings.num_classes,
            "classes": settings.vertebrae_classes,
            "backbone": settings.model_backbone,
            "device": settings.device,
            "confidence_threshold": settings.confidence_threshold,
            "nms_threshold": settings.nms_threshold
        }


def get_model() -> VertebraeSegmentationModel:
    """Get singleton instance of the vertebrae segmentation model."""
    return VertebraeSegmentationModel()
