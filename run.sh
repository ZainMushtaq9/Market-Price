#!/bin/bash

# Market-Price Startup Script

echo "🌾 Starting Market-Price Platform..."
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if requirements are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing..."
    pip3 install -r requirements.txt
fi

echo "✓ Dependencies installed"

# Check if database exists
if [ -f "database.db" ]; then
    echo "✓ Database found"
else
    echo "ℹ️  Database will be created on first run"
fi

echo ""
echo "🚀 Starting FastAPI server..."
echo "📍 Access the platform at: http://localhost:8000"
echo "📊 API documentation at: http://localhost:8000/docs"
echo "❤️  Health check at: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=================================="
echo ""

# Run the application
python3 app.py
