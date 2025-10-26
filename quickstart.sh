#!/bin/bash

# Quick Start Script for Embedding Encoder Training

set -e  # Exit on error

echo "======================================="
echo "Embedding Encoder Training Quick Start"
echo "======================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements-training.txt

echo ""
echo "Dependencies installed successfully!"
echo ""

# Check if data exists
if [ ! -d "data/embeddings" ] || [ -z "$(ls -A data/embeddings/*.csv 2>/dev/null)" ]; then
    echo "No training data found. Creating example data..."
    python prepare_data.py example --output_dir data/embeddings --num_samples 100 --vector_dim 768
    echo ""
fi

# Test data loading
echo "Testing data loading..."
python encoder_dataset.py --data_dir data/embeddings --split train
echo ""

# Ask user if they want to start training
read -p "Do you want to start training now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training..."
    echo ""
    python train_encoder.py --config config/train_config.yaml
else
    echo ""
    echo "Setup complete! To start training later, run:"
    echo "  source venv/bin/activate"
    echo "  python train_encoder.py --config config/train_config.yaml"
    echo ""
    echo "To monitor training with TensorBoard:"
    echo "  tensorboard --logdir outputs/encoder_training/logs"
fi
