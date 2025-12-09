#!/bin/bash

echo "========================================="
echo "Setting up dependencies for Qwen3 ROS2"
echo "========================================="

# Install pyzmq in conda environment
echo ""
echo "Installing pyzmq in conda environment (yeseul)..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yeseul
pip install pyzmq
conda deactivate

# Install pyzmq in system Python
echo ""
echo "Installing pyzmq in system Python..."
python3 -m pip install --user pyzmq

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run the system:"
echo "1. Terminal 1 (conda env): conda activate yeseul && python3 qwen3_model_server.py"
echo "2. Terminal 2 (system): python3 qwen3_ros2_client_node.py"
echo ""

