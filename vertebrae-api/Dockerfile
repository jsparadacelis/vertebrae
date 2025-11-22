# =========================
# === BUILDER STAGE =======
# =========================
FROM python:3.10-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=0 \
    TORCH_CUDA_ARCH_LIST=""

WORKDIR /app

# --- Install build tools for detectron2 & pycocotools ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Install base deps including NUMPY 1.x FIRST ---
RUN pip install "numpy>=1.21,<2.0"

# --- Install PyTorch CPU-only ---
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# --- Install pycocotools BEFORE detectron2 ---
RUN pip install pycocotools

# --- Install Detectron2 from GitHub ---
RUN pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'

# --- Install server deps (FastAPI, uvicorn, etc.) ---
RUN pip install \
    fastapi \
    python-multipart \
    uvicorn[standard] \
    pydantic \
    opencv-python \
    pydantic-settings \
    python-dotenv \
    boto3 \
    ultralytics

# --- FINAL FIX: force reinstall correct NumPy (ensures no package upgraded it) ---
RUN pip install --force-reinstall "numpy>=1.21,<2.0"


# =========================
# === FINAL RUNTIME =======
# =========================
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# --- Install only lightweight runtime libs ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --- Copy everything from builder ---
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# --- Copy your application code ---
COPY app/ ./app/

# --- Model cache directory ---
RUN mkdir -p /tmp/model_cache && chmod 777 /tmp/model_cache

# --- Create non-root user ---
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /tmp/model_cache

USER appuser

EXPOSE 8000

# --- Healthcheck ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# --- Start API ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]