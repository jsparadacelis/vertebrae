# Installation Guide

## Prerequisites

- Python 3.10+
- pip (latest version)
- Git
- Ubuntu/Linux recommended

## Installation Methods

### Method 1: Automated Installation Script (Recommended)

```bash
./install.sh
```

This script installs all dependencies in the correct order.

### Method 2: Manual Installation

Follow these steps **in order**:

```bash
# 1. Upgrade pip
pip install --upgrade pip setuptools wheel

# 2. Install PyTorch (MUST be before Detectron2)
pip install torch==2.1.2 torchvision==0.16.2

# 3. Install Detectron2 (requires --no-build-isolation)
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 4. Install computer vision dependencies
pip install opencv-python==4.9.0.80
pip install pycocotools==2.0.7

# 5. Install remaining dependencies
pip install fastapi==0.109.0
pip install uvicorn[standard]==0.27.0
pip install python-multipart==0.0.6
pip install boto3==1.34.34
pip install botocore==1.34.34
pip install pydantic==2.5.3
pip install pydantic-settings==2.1.0
pip install python-dotenv==1.0.0
pip install ultralytics==8.1.0
```

### Method 3: Docker (Production)

```bash
# Build image
docker build -t vertebrae-api .

# Run with docker-compose
docker-compose up -d
```

## Common Installation Issues

### Issue 1: Detectron2 Build Fails

**Error**: `error: metadata-generation-failed`

**Solution**: Use `--no-build-isolation` flag:
```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

### Issue 2: PyTorch Not Found During Detectron2 Install

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install PyTorch **before** Detectron2:
```bash
pip install torch==2.1.2 torchvision==0.16.2
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

### Issue 3: OpenCV Import Errors

**Error**: `ImportError: libGL.so.1`

**Solution** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Issue 4: Missing Build Tools

**Error**: `error: command 'gcc' failed`

**Solution** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev git
```

## Verification

After installation, verify all packages:

```bash
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import detectron2; print('Detectron2: OK')"
python -c "import ultralytics; print('Ultralytics: OK')"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your AWS credentials:
```bash
nano .env
```

Required variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET`
- `YOLO_MODEL_KEY`
- `MASKRCNN_MODEL_KEY`

## Running the API

### Local Development

```bash
# Option 1: Using Python
python -m app.main

# Option 2: Using Uvicorn with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Using Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Using Docker
docker-compose up -d
```

## Testing the Installation

```bash
# Check health
curl http://localhost:8000/health

# Check available models
curl http://localhost:8000/models

# Test prediction
curl -X POST "http://localhost:8000/predict?model=yolo" \
  -F "file=@test_image.jpg"
```

## System Requirements

### Minimum
- CPU: 2 cores
- RAM: 4 GB
- Disk: 5 GB free space

### Recommended
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 10+ GB free space
- GPU: Optional (for faster inference)

## Notes

- **Installation Order Matters**: PyTorch must be installed before Detectron2
- **Detectron2 Flag**: Always use `--no-build-isolation` when installing Detectron2
- **CPU vs GPU**: Default installation uses CPU-only PyTorch for compatibility
- **Model Download**: Models are downloaded from S3 on first API startup (requires AWS credentials)
