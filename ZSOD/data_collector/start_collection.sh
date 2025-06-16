#!/bin/bash

echo "=== YOLO Dataset Collector Starter ==="
echo

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo "Warning: ROS2 environment not sourced!"
    echo "Please run: source /opt/ros/humble/setup.bash"
    echo
fi

# Show menu
echo "Select collection mode:"
echo "1) High Quality Collection (conf: 0.8, interval: 3s)"
echo "2) Balanced Collection (conf: 0.6, interval: 2s) [DEFAULT]"
echo "3) Fast Collection (conf: 0.5, interval: 1s)"
echo "4) Custom Settings"
echo "5) Show current ROS2 image topics"
echo

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo "Starting High Quality Collection..."
        python3 run_dataset_collector.py \
            --conf_threshold 0.8 \
            --iou_threshold 0.3 \
            --collection_interval 3.0 \
            --min_detections 2
        ;;
    2)
        echo "Starting Balanced Collection..."
        python3 run_dataset_collector.py
        ;;
    3)
        echo "Starting Fast Collection..."
        python3 run_dataset_collector.py \
            --conf_threshold 0.5 \
            --collection_interval 1.0 \
            --max_detections 100
        ;;
    4)
        echo "Custom Settings:"
        read -p "Confidence threshold (0.1-1.0): " conf
        read -p "Collection interval (seconds): " interval
        read -p "Image topic (default: /stereo_image_color): " topic
        
        if [ -z "$topic" ]; then
            topic="/stereo_image_color"
        fi
        
        echo "Starting with conf=$conf, interval=${interval}s, topic=$topic"
        python3 run_dataset_collector.py \
            --conf_threshold $conf \
            --collection_interval $interval \
            --image_topic $topic
        ;;
    5)
        echo "Available image topics:"
        ros2 topic list | grep -E "(image|cam)" || echo "No image topics found"
        echo
        echo "Re-run this script to start collection."
        ;;
    *)
        echo "Invalid choice. Starting with default settings..."
        python3 run_dataset_collector.py
        ;;
esac 