#!/bin/bash

# Emotion-Aware Storyteller Setup Script
# Automates the installation and setup process

set -e

echo "🎭 Emotion-Aware Storyteller - Setup Script"
echo "==========================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for Ollama
echo ""
echo "🔍 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "   ✓ Ollama is installed"
    echo ""
    echo "📥 Pulling Llama 3 model (this may take a while)..."
    ollama pull llama3
else
    echo "   ⚠️  Ollama not found"
    echo "   Install from: https://ollama.com/download"
    echo "   Or use --llm-backend template for instant responses"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "   1. Activate environment: source venv/bin/activate"
echo "   2. Run: python src/app.py"
echo ""
echo "📖 See README.md for more options and documentation"
