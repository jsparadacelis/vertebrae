"""Abstract base class for segmentation models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

import numpy as np


class BaseSegmentationModel(ABC):
    """Abstract base class for vertebrae segmentation models."""

    def __init__(self):
        """Initialize the model."""
        self._model_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the segmentation model from S3 or local cache."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an image.

        Args:
            image: Input image as numpy array in BGR format.

        Returns:
            Dictionary containing predictions with bounding boxes, masks, scores, and classes.
        """
        pass

    @abstractmethod
    def format_detections(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format raw predictions into structured detection objects.

        Args:
            predictions: Raw predictions from model.

        Returns:
            List of formatted detection dictionaries.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
