"""Utility functions for S3 and image processing."""

import io
import logging
from pathlib import Path
from typing import List

import boto3
import cv2
import numpy as np
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def download_model_from_s3(model_key: str, cache_filename: str) -> Path:
    """
    Download a model from S3 if not already cached locally.

    Args:
        model_key: S3 key for the model file.
        cache_filename: Local filename to cache the model as.

    Returns:
        Path to the downloaded model file.

    Raises:
        Exception: If S3 download fails.
    """
    model_path = settings.get_model_cache_path(cache_filename)

    if model_path.exists():
        logger.info(f"Model already cached at {model_path}")
        return model_path

    logger.info(f"Downloading model from s3://{settings.s3_bucket}/{model_key}")

    try:
        # Use default AWS credentials or explicit from settings
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
        else:
            # Use default credentials (AWS CLI, IAM role, etc.)
            s3_client = boto3.client('s3', region_name=settings.aws_region)

        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            settings.s3_bucket,
            model_key,
            str(model_path)
        )

        logger.info(f"Model downloaded successfully to {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Failed to download model from S3: {e}")
        raise



def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load an image from bytes and convert to BGR format for OpenCV/Detectron2.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Image as numpy array in BGR format.

    Raises:
        ValueError: If image cannot be loaded.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array and then to BGR for OpenCV/Detectron2
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        return image_bgr

    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise ValueError(f"Invalid image format: {e}")


def draw_predictions_on_image(
    image: np.ndarray,
    boxes: List[List[float]],
    masks: List[np.ndarray],
    scores: List[float],
    classes: List[str],
    score_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes, masks, and labels on an image.

    Args:
        image: Input image in BGR format.
        boxes: List of bounding boxes [x1, y1, x2, y2].
        masks: List of binary masks.
        scores: List of confidence scores.
        classes: List of class labels.
        score_threshold: Minimum score to display.

    Returns:
        Annotated image in BGR format.
    """
    annotated = image.copy()
    img_height, img_width = image.shape[:2]

    # Generate random colors for each class
    np.random.seed(42)
    colors = {
        cls: tuple(map(int, np.random.randint(0, 255, 3)))
        for cls in set(classes)
    }

    for box, mask, score, cls in zip(boxes, masks, scores, classes):
        if score < score_threshold:
            continue

        color = colors[cls]

        # Resize mask to match image dimensions if needed
        if mask.shape[:2] != (img_height, img_width):
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (img_width, img_height),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask.astype(np.uint8)

        # Draw filled mask with transparency
        mask_overlay = annotated.copy()
        mask_overlay[mask_resized > 0] = color
        annotated = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)

        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label with score
        label = f"{cls}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return annotated


def image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """
    Convert numpy image array to bytes.

    Args:
        image: Image as numpy array in BGR format.
        format: Output format (PNG, JPEG, etc.).

    Returns:
        Image as bytes.
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)

    return buffer.getvalue()
