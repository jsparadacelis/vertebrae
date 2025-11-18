"""Model factory for managing multiple segmentation models."""

import logging
from enum import Enum
from typing import Dict, Optional

from app.models.base import BaseSegmentationModel
from app.models.maskrcnn import MaskRCNNModel
from app.models.yolo import YOLOModel
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelType(str, Enum):
    """Supported model types."""
    YOLO = "yolo"
    MASKRCNN = "maskrcnn"


class ModelFactory:
    """Factory class for creating and managing segmentation models."""

    _instances: Dict[ModelType, BaseSegmentationModel] = {}
    _initialized: bool = False

    @classmethod
    def initialize_models(cls) -> None:
        """Initialize all models on startup."""
        if cls._initialized:
            logger.info("Models already initialized")
            return

        logger.info("Initializing all models...")

        try:
            # Initialize YOLO model
            logger.info("Loading YOLO model...")
            yolo_model = YOLOModel()
            yolo_model.load_model()
            cls._instances[ModelType.YOLO] = yolo_model
            logger.info("YOLO model initialized successfully")

            # Initialize Mask R-CNN model
            logger.info("Loading Mask R-CNN model...")
            maskrcnn_model = MaskRCNNModel()
            maskrcnn_model.load_model()
            cls._instances[ModelType.MASKRCNN] = maskrcnn_model
            logger.info("Mask R-CNN model initialized successfully")

            cls._initialized = True
            logger.info("All models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    @classmethod
    def get_model(cls, model_type: Optional[ModelType] = None) -> BaseSegmentationModel:
        """
        Get a model instance by type.

        Args:
            model_type: Type of model to retrieve. If None, uses default from settings.

        Returns:
            Initialized model instance.

        Raises:
            ValueError: If model type is invalid or not initialized.
        """
        # Use default model if not specified
        if model_type is None:
            model_type = ModelType(settings.default_model)

        # Ensure models are initialized
        if not cls._initialized:
            cls.initialize_models()

        # Get model instance
        if model_type not in cls._instances:
            raise ValueError(f"Model type '{model_type}' not available")

        model = cls._instances[model_type]

        if not model.is_loaded():
            raise RuntimeError(f"Model '{model_type}' is not loaded")

        return model

    @classmethod
    def get_all_models_info(cls) -> Dict[str, dict]:
        """Get information about all loaded models."""
        if not cls._initialized:
            return {}

        models_info = {}
        for model_type, model in cls._instances.items():
            if model.is_loaded():
                models_info[model_type.value] = model.get_model_info()

        return models_info

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if models are initialized."""
        return cls._initialized


def get_model(model_type: Optional[ModelType] = None) -> BaseSegmentationModel:
    """
    Convenience function to get a model instance.

    Args:
        model_type: Type of model to retrieve. If None, uses default from settings.

    Returns:
        Initialized model instance.
    """
    return ModelFactory.get_model(model_type)


def initialize_all_models() -> None:
    """Convenience function to initialize all models."""
    ModelFactory.initialize_models()


def get_all_models_info() -> Dict[str, dict]:
    """Convenience function to get all models info."""
    return ModelFactory.get_all_models_info()
