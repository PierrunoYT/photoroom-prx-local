#!/bin/bash

echo "========================================"
echo "PRX-1024 Web UI Setup"
echo "========================================"
echo ""

echo "Upgrading pip..."
python -m pip install --upgrade pip
echo ""

echo "Installing PyTorch 2.6+ with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo ""

echo "Installing diffusers from GitHub (required for PRX support)..."
pip install --upgrade git+https://github.com/huggingface/diffusers.git
echo ""

echo "Upgrading transformers (required for T5Gemma support)..."
pip install --upgrade transformers
echo ""

echo "Installing other requirements..."
pip install gradio accelerate safetensors Pillow sentencepiece
echo ""

echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To run the application, use:"
echo "  python app.py"
echo ""

