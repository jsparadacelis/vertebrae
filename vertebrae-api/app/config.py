"""Configuration management for the vertebrae segmentation API."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # AWS Configuration
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    # S3 Configuration
    s3_bucket: str = "vertebrae-artifacts"
    model_cache_dir: Path = Path("/tmp/model_cache")

    # Model Selection
    default_model: str = "yolo"  # Options: "yolo" or "maskrcnn"

    # YOLO Model Configuration
    yolo_model_key: str = "yolo_best.pt"

    # Mask R-CNN Model Configuration
    maskrcnn_model_key: str = "model_final.pth"
    maskrcnn_backbone: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Model Inference Configuration
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100
    device: str = "cpu"

    # Vertebrae Classes (T1-T12, L1-L5)
    vertebrae_classes: List[str] = [
        "T1", "T2", "T3", "T4", "T5", "T6",
        "T7", "T8", "T9", "T10", "T11", "T12",
        "L1", "L2", "L3", "L4", "L5"
    ]
    num_classes: int = 17

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def get_model_cache_path(self, filename: str) -> Path:
        """Get the local path where a model should be cached."""
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.model_cache_dir / filename


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
