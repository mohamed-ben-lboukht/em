#!/bin/bash

echo "🐅 My Tiger - Emotion Detection Application"
echo "============================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing/updating dependencies..."
source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1

# Start the application
echo "🚀 Starting My Tiger application..."
echo "   Open your browser and go to: http://localhost:8000"
echo "   Press Ctrl+C to stop the application"
echo ""


python run.py 