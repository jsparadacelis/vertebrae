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

# Install PyTorch CPU-only (smaller) - version 2.2.0+ supports NumPy 2.x
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Detectron2 CPU-only with minimal dependencies
ENV FORCE_CUDA="0"
ENV TORCH_CUDA_ARCH_LIST=""

# Install remaining dependencies (PyTorch 2.2.0+ handles NumPy 2.x automatically)
RUN pip install --no-cache-dir \
    opencv-python \
    pycocotools \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    boto3 \
    botocore \
    pydantic \
    pydantic-settings \
    python-dotenv \
    ultralytics

RUN pip install --no-cache-dir --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'


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
