# Multi-stage build for vertebrae segmentation API with dual model support
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies in correct order
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version (must be before Detectron2)
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Install Detectron2 with --no-build-isolation flag
RUN pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Install computer vision dependencies
RUN pip install opencv-python==4.9.0.80 pycocotools==2.0.7

# Install remaining dependencies from requirements.txt
RUN pip install fastapi==0.109.0 uvicorn[standard]==0.27.0 python-multipart==0.0.6 \
    boto3==1.34.34 botocore==1.34.34 \
    pydantic==2.5.3 pydantic-settings==2.1.0 python-dotenv==1.0.0 \
    ultralytics==8.1.0

# Copy application code
COPY app/ ./app/

# Create model cache directory with proper permissions
RUN mkdir -p /tmp/model_cache && chmod 777 /tmp/model_cache

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/model_cache

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
