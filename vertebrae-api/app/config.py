from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    s3_bucket: str = "vertebrae-artifacts"
    model_s3_key: str = "model_final.pth"
    num_classes: int = 17
    confidence_threshold: float = 0.7
    model_cache_dir: str = "/tmp/models"
    device: str = "cpu"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
