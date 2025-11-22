# ============================
# Builder Image
# ============================
FROM python:3.10-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=0 \
    TORCH_CUDA_ARCH_LIST=""

WORKDIR /app

# ----------------------------
# Install build tools
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python Dependencies
# ----------------------------

# Create a requirements file inline for reproducibility
# (You can replace this with COPY requirements.txt if you prefer)
RUN echo "\
numpy>=1.21.0,<2.0\n\
pycocotools\n\
torch==2.1.2\n\
torchvision==0.16.2\n\
uvicorn[standard]\n\
" > /tmp/requirements.txt

# Install base deps — detectron2 comes afterwards
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install detectron2 (CPU-only)
RUN pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'

# ----------------------------
# Copy just the application
# ----------------------------
COPY app/ ./app/

# ============================
# Runtime Image
# ============================
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only what is needed from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                   /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/app ./app

# Create writable model cache
RUN mkdir -p /tmp/model_cache && chmod 777 /tmp/model_cache

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /tmp/model_cache
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
