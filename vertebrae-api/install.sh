#!/bin/bash
# Installation script for vertebrae-api
# This script installs dependencies in the correct order

set -e

echo "🔧 Installing Vertebrae Segmentation API dependencies..."
echo ""

# Step 1: Upgrade pip
echo "📦 Step 1/6: Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
echo "✅ Done"
echo ""

# Step 2: Install PyTorch (must be installed before Detectron2)
echo "📦 Step 2/6: Installing PyTorch and TorchVision..."
pip install torch==2.1.2 torchvision==0.16.2
echo "✅ Done"
echo ""

# Step 3: Install Detectron2 (requires --no-build-isolation flag)
echo "📦 Step 3/6: Installing Detectron2..."
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
echo "✅ Done"
echo ""

# Step 4: Install computer vision dependencies
echo "📦 Step 4/6: Installing OpenCV and COCO tools..."
pip install opencv-python==4.9.0.80
pip install pycocotools==2.0.7
echo "✅ Done"
echo ""

# Step 5: Install remaining dependencies from requirements.txt
echo "📦 Step 5/6: Installing remaining dependencies..."
pip install fastapi==0.109.0
pip install uvicorn[standard]==0.27.0
pip install python-multipart==0.0.6
pip install boto3==1.34.34
pip install botocore==1.34.34
pip install pydantic==2.5.3
pip install pydantic-settings==2.1.0
pip install python-dotenv==1.0.0
pip install ultralytics==8.1.0
echo "✅ Done"
echo ""

# Step 6: Verify installation
echo "📦 Step 6/6: Verifying installation..."
python -c "import fastapi; print('✅ FastAPI:', fastapi.__version__)"
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import detectron2; print('✅ Detectron2: OK')"
python -c "import ultralytics; print('✅ Ultralytics: OK')"
python -c "import cv2; print('✅ OpenCV:', cv2.__version__)"
python -c "import boto3; print('✅ Boto3: OK')"
echo ""

echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your AWS credentials"
echo "2. Run the API: python -m app.main"
echo "   or with uvicorn: uvicorn app.main:app --reload"
