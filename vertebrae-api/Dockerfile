# Optimized Dockerfile for vertebrae segmentation API (CPU-only, minimal size)
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies (will be removed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install NumPy 1.x FIRST - this is critical
RUN pip install --no-cache-dir "numpy>=1.21.0,<2.0"

# Install PyTorch CPU-only (specify version that works with NumPy 1.x)
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Install Detectron2 CPU-only with minimal dependencies
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

# Install pycocotools BEFORE detectron2
RUN pip install --no-cache-dir pycocotools

# Install detectron2
RUN pip install --no-cache-dir --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# Install remaining dependencies (avoiding numpy upgrades)
RUN pip install --no-cache-dir --no-deps \
    opencv-python-headless \
    fastapi \
    python-multipart \
    boto3 \
    botocore \
    pydantic \
    pydantic-settings \
    python-dotenv

# Install uvicorn separately with its dependencies
RUN pip install --no-cache-dir uvicorn[standard]

# Install ultralytics WITHOUT allowing it to upgrade numpy
RUN pip install --no-cache-dir --upgrade --no-deps ultralytics && \
    pip install --no-cache-dir \
    matplotlib \
    pillow \
    pyyaml \
    requests \
    scipy \
    tqdm \
    psutil \
    py-cpuinfo \
    thop \
    pandas \
    seaborn

# Force NumPy back to 1.x in case something upgraded it
RUN pip install --no-cache-dir --force-reinstall "numpy>=1.21.0,<2.0"


# Final stage - minimal runtime image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install only runtime dependencies (much smaller)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/

# Create model cache directory
RUN mkdir -p /tmp/model_cache && chmod 777 /tmp/model_cache

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/model_cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]