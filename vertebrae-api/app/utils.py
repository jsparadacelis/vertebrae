"""Utility functions for S3, image processing, and mask encoding."""

import io
import logging
from pathlib import Path
from typing import Dict, List, Any

import boto3
import cv2
import os
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def download_model_from_s3():
    bucket = os.getenv('S3_BUCKET', 'vertebrae-artifacts')
    key = os.getenv('MODEL_S3_KEY', 'model_final.pth')
    cache_dir = os.getenv('MODEL_CACHE_DIR', '/tmp/models')
    
    logger.info(f"Downloading from s3://{bucket}/{key}")
    
    # Use default credentials (same as AWS CLI) - DON'T pass explicit credentials
    s3_client = boto3.client('s3')
    
    # Prepare local path
    local_path = Path(cache_dir) / 'model_final.pth'
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if local_path.exists():
        logger.info(f"Model already cached at {local_path}")
        return str(local_path)
    
    try:
        logger.info("Downloading model from S3...")
        s3_client.download_file(bucket, key, str(local_path))
        logger.info(f"✅ Model downloaded to {local_path}")
        return str(local_path)
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
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


def encode_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask to RLE (Run-Length Encoding) format.

    Args:
        mask: Binary mask as 2D numpy array (H, W).

    Returns:
        Dictionary with RLE encoding compatible with COCO format.
    """
    # Ensure mask is in correct format (Fortran order, uint8)
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask)

    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')

    return rle


def decode_rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode RLE format back to binary mask.

    Args:
        rle: RLE dictionary with 'size' and 'counts'.

    Returns:
        Binary mask as 2D numpy array.
    """
    # Convert string back to bytes if necessary
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')

    mask = mask_util.decode(rle)
    return mask


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

        # Draw filled mask with transparency
        mask_overlay = annotated.copy()
        mask_overlay[mask > 0] = color
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
