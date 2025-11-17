from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # S3 Model Configuration
    s3_bucket: str = "vertebrae-artifacts"
    s3_model_key: str = "model_final.pth"
    model_cache_dir: Path = Path("/tmp/model_cache")

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Model Configuration
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    max_detections: int = 100

    # Vertebrae Classes (T1-T12, L1-L5)
    vertebrae_classes: List[str] = [
        "T1", "T2", "T3", "T4", "T5", "T6",
        "T7", "T8", "T9", "T10", "T11", "T12",
        "L1", "L2", "L3", "L4", "L5"
    ]
    num_classes: int = 17

    # Detectron2 Model Configuration
    model_backbone: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    device: str = "cpu"

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def get_model_path(self) -> Path:
        """Get the local path where the model should be cached."""
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        return self.model_cache_dir / self.s3_model_key


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
