# Vertebrae Segmentation API

FastAPI-based REST API for vertebrae segmentation using Mask R-CNN with Detectron2. Detects and segments 17 vertebrae classes (T1-T12, L1-L5) from X-ray or medical images.

## Features

- **Mask R-CNN** with ResNet-50 FPN backbone
- **17 vertebrae classes**: T1-T12 (thoracic), L1-L5 (lumbar)
- **CPU inference** for broad compatibility
- **S3 model storage** with automatic download and caching
- **Multiple endpoints**: JSON predictions, visualized images, health checks
- **Docker deployment** for easy deployment
- **RLE mask encoding** for efficient JSON serialization

## Architecture

```
vertebrae-api/
├── app/
│   ├── main.py          # FastAPI app with endpoints
│   ├── config.py        # Configuration management
│   ├── model.py         # Detectron2 model wrapper
│   ├── schemas.py       # Pydantic models
│   └── utils.py         # S3, image processing, mask encoding
├── tests/
│   └── test_api.py      # API endpoint tests
├── Dockerfile           # Container definition
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (create from .env.example)
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- AWS credentials with S3 access
- Model file in S3: `s3://vertebrae-artifacts/model_final.pth`

## Installation

### 1. Clone and Setup

```bash
cd vertebrae-api
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` with your AWS credentials and configuration:

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# S3 Model Configuration
S3_BUCKET=vertebrae-artifacts
S3_MODEL_KEY=model_final.pth
```

### 3. Option A: Docker Deployment (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or with Docker directly
docker build -t vertebrae-api .
docker run -p 8000:8000 --env-file .env vertebrae-api
```

### 3. Option B: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API
python -m app.main
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Information
```bash
GET /
```

Returns API information and available endpoints.

### 2. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/tmp/model_cache/model_final.pth"
}
```

### 3. Model Information
```bash
GET /model-info
```

**Response:**
```json
{
  "model_name": "Mask R-CNN",
  "num_classes": 17,
  "classes": ["T1", "T2", ..., "L5"],
  "backbone": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
  "device": "cpu",
  "confidence_threshold": 0.5,
  "nms_threshold": 0.5
}
```

### 4. Predict (JSON Response)
```bash
POST /predict
```

Upload an image and receive JSON predictions with bounding boxes, masks (RLE), scores, and classes.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@spine_xray.jpg"
```

**Response:**
```json
{
  "detections": [
    {
      "bbox": {
        "x1": 245.3,
        "y1": 102.7,
        "x2": 312.8,
        "y2": 156.4
      },
      "mask": {
        "size": [512, 512],
        "counts": "RLE_encoded_string..."
      },
      "score": 0.95,
      "class_name": "T1",
      "class_id": 0
    },
    ...
  ],
  "num_detections": 14,
  "image_shape": [512, 512, 3],
  "processing_time_ms": 234.56
}
```

### 5. Predict with Visualization
```bash
POST /predict/visualize
```

Upload an image and receive an annotated PNG image with bounding boxes, masks, and labels.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/visualize" \
  -F "file=@spine_xray.jpg" \
  --output annotated.png
```

**Response:**
- Content-Type: `image/png`
- Headers:
  - `X-Num-Detections`: Number of detected vertebrae
  - `X-Processing-Time-Ms`: Inference time in milliseconds

## Usage Examples

### Python Client

```python
import requests

# Predict with JSON response
with open("spine_xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    predictions = response.json()

print(f"Found {predictions['num_detections']} vertebrae")
for det in predictions['detections']:
    print(f"{det['class_name']}: {det['score']:.2f}")

# Get visualized result
with open("spine_xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/visualize",
        files={"file": f}
    )
    with open("annotated.png", "wb") as out:
        out.write(response.content)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info

# Predict
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg" \
  -o predictions.json

# Visualize
curl -X POST http://localhost:8000/predict/visualize \
  -F "file=@image.jpg" \
  -o annotated.png
```

## Model Configuration

The model is configured in `app/config.py` and can be customized via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.5 | Minimum confidence score for detections |
| `NMS_THRESHOLD` | 0.5 | Non-maximum suppression threshold |
| `MAX_DETECTIONS` | 100 | Maximum number of detections per image |
| `MODEL_CACHE_DIR` | `/tmp/model_cache` | Directory for caching downloaded model |

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov httpx

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Mask Encoding

Segmentation masks are encoded using Run-Length Encoding (RLE) in COCO format for efficient JSON serialization:

```python
{
  "size": [height, width],
  "counts": "encoded_string"
}
```

To decode masks in Python:

```python
from pycocotools import mask as mask_util
import numpy as np

def decode_mask(rle_dict):
    # Convert string to bytes if needed
    if isinstance(rle_dict['counts'], str):
        rle_dict['counts'] = rle_dict['counts'].encode('utf-8')
    return mask_util.decode(rle_dict)

# Usage
mask_binary = decode_mask(detection['mask'])
```

## Performance

- **CPU Inference**: ~200-500ms per image (512x512)
- **Model Size**: ~170MB
- **Memory Usage**: ~2-3GB RAM

For faster inference, modify `DEVICE=cuda` in `.env` and use a GPU-enabled Docker image.

## Troubleshooting

### Model Download Fails
- Verify AWS credentials in `.env`
- Check S3 bucket permissions
- Ensure model exists at `s3://vertebrae-artifacts/model_final.pth`

### Out of Memory
- Reduce `MAX_DETECTIONS`
- Use smaller input images
- Increase Docker memory limit

### Slow Inference
- CPU inference is slower than GPU
- Consider using smaller images
- For production, use GPU-enabled deployment

## Deployment

### Production Recommendations

1. **Use GPU**: Set `DEVICE=cuda` for faster inference
2. **Load Balancing**: Deploy multiple instances behind a load balancer
3. **Monitoring**: Add logging and monitoring (Prometheus, Grafana)
4. **Secrets Management**: Use AWS Secrets Manager or similar for credentials
5. **Rate Limiting**: Implement rate limiting for API endpoints
6. **HTTPS**: Deploy behind a reverse proxy with SSL (nginx, Traefik)

### Example nginx Configuration

```nginx
server {
    listen 80;
    server_name vertebrae-api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;
    }
}
```

## License

This project is part of the vertebrae segmentation system.

## Support

For issues or questions, please contact the development team or open an issue in the repository.
