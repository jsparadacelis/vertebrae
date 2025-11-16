#!/bin/bash
# Quick start script for local development

set -e

echo "🚀 Starting Vertebrae Segmentation API..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your AWS credentials before running again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run the API
echo "✅ Starting API server..."
echo "📍 API will be available at: http://localhost:8000"
echo "📚 API docs available at: http://localhost:8000/docs"
echo ""

python3 -m app.main
