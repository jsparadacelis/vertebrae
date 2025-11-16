"""Basic API endpoint tests."""

import io
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.config import get_settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create a sample image as bytes."""
    # Create a simple RGB image
    image = Image.new('RGB', (512, 512), color='white')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def mock_model_predictions():
    """Mock model predictions."""
    return {
        "boxes": [[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]],
        "scores": [0.95, 0.87],
        "classes": [0, 5],
        "masks": np.array([
            np.random.randint(0, 2, (512, 512), dtype=np.uint8),
            np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        ]),
        "num_detections": 2,
        "processing_time_ms": 123.45,
        "image_shape": [512, 512, 3]
    }


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["name"] == "Vertebrae Segmentation API"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_returns_status(self, client):
        """Test that health endpoint returns status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "unhealthy"]


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info_returns_configuration(self, client):
        """Test that model info endpoint returns configuration."""
        response = client.get("/model-info")
        assert response.status_code == 200

        data = response.json()
        assert "model_name" in data
        assert "num_classes" in data
        assert "classes" in data
        assert "backbone" in data
        assert data["num_classes"] == 17
        assert len(data["classes"]) == 17


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_requires_file(self, client):
        """Test that predict endpoint requires a file."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error

    def test_predict_rejects_empty_file(self, client):
        """Test that predict endpoint rejects empty files."""
        response = client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")}
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    @patch('app.main.get_model')
    def test_predict_returns_detections(self, mock_get_model, client, sample_image_bytes, mock_model_predictions):
        """Test successful prediction with mock model."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_model_predictions
        mock_model.format_detections.return_value = [
            {
                "bbox": {"x1": 100.0, "y1": 100.0, "x2": 200.0, "y2": 200.0},
                "mask": {"size": [512, 512], "counts": "mock_rle_string"},
                "score": 0.95,
                "class_name": "T1",
                "class_id": 0
            },
            {
                "bbox": {"x1": 300.0, "y1": 300.0, "x2": 400.0, "y2": 400.0},
                "mask": {"size": [512, 512], "counts": "mock_rle_string"},
                "score": 0.87,
                "class_name": "T6",
                "class_id": 5
            }
        ]
        mock_get_model.return_value = mock_model

        # Send request
        response = client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()

        assert "detections" in data
        assert "num_detections" in data
        assert "image_shape" in data
        assert "processing_time_ms" in data

        assert data["num_detections"] == 2
        assert len(data["detections"]) == 2
        assert data["detections"][0]["class_name"] == "T1"
        assert data["detections"][1]["class_name"] == "T6"


class TestVisualizeEndpoint:
    """Tests for visualization endpoint."""

    def test_visualize_requires_file(self, client):
        """Test that visualize endpoint requires a file."""
        response = client.post("/predict/visualize")
        assert response.status_code == 422  # Validation error

    def test_visualize_rejects_empty_file(self, client):
        """Test that visualize endpoint rejects empty files."""
        response = client.post(
            "/predict/visualize",
            files={"file": ("empty.png", b"", "image/png")}
        )
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]

    @patch('app.main.get_model')
    def test_visualize_returns_image(self, mock_get_model, client, sample_image_bytes, mock_model_predictions):
        """Test successful visualization with mock model."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_model_predictions
        mock_get_model.return_value = mock_model

        # Send request
        response = client.post(
            "/predict/visualize",
            files={"file": ("test.png", sample_image_bytes, "image/png")}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert "X-Num-Detections" in response.headers
        assert "X-Processing-Time-Ms" in response.headers
        assert int(response.headers["X-Num-Detections"]) == 2
