#!/usr/bin/env python3

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='YOLO Dataset Collector Configuration')
    
    # Collection parameters
    parser.add_argument('--conf_threshold', type=float, default=0.6,
                       help='Confidence threshold for detection (default: 0.6)')
    parser.add_argument('--iou_threshold', type=float, default=0.4,
                       help='IOU threshold for NMS (default: 0.4)')
    parser.add_argument('--collection_interval', type=float, default=2.0,
                       help='Collection interval in seconds (default: 2.0)')
    parser.add_argument('--min_detections', type=int, default=1,
                       help='Minimum detections required to save (default: 1)')
    parser.add_argument('--max_detections', type=int, default=50,
                       help='Maximum detections per image (default: 50)')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='~/yolo_dataset',
                       help='Dataset save path (default: ~/yolo_dataset)')
    parser.add_argument('--image_topic', type=str, default='/stereo_image_color',
                       help='Input image topic (default: /stereo_image_color)')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='yoloe-11m-seg.pt',
                       help='YOLOE model path (default: yoloe-11m-seg.pt)')
    
    # Test mode
    parser.add_argument('--test_mode', action='store_true',
                       help='Enable test mode (detection only, no data collection)')
    
    args = parser.parse_args()
    
    # Set environment variables for the collector
    os.environ['COLLECTOR_CONF_THRESHOLD'] = str(args.conf_threshold)
    os.environ['COLLECTOR_IOU_THRESHOLD'] = str(args.iou_threshold)
    os.environ['COLLECTOR_INTERVAL'] = str(args.collection_interval)
    os.environ['COLLECTOR_MIN_DETECTIONS'] = str(args.min_detections)
    os.environ['COLLECTOR_MAX_DETECTIONS'] = str(args.max_detections)
    os.environ['COLLECTOR_DATASET_PATH'] = args.dataset_path
    os.environ['COLLECTOR_IMAGE_TOPIC'] = args.image_topic
    os.environ['COLLECTOR_MODEL_PATH'] = args.model_path
    os.environ['COLLECTOR_TEST_MODE'] = str(args.test_mode)
    
    if args.test_mode:
        print("=== YOLO Dataset Collector - TEST MODE ===")
        print("🔍 Detection Only Mode (No Data Collection)")
    else:
        print("=== YOLO Dataset Collector Configuration ===")
    
    print(f"Confidence Threshold: {args.conf_threshold}")
    print(f"IOU Threshold: {args.iou_threshold}")
    print(f"Collection Interval: {args.collection_interval}s")
    
    if not args.test_mode:
        print(f"Min Detections: {args.min_detections}")
        print(f"Max Detections: {args.max_detections}")
        print(f"Dataset Path: {args.dataset_path}")
    
    print(f"Image Topic: {args.image_topic}")
    print(f"Model Path: {args.model_path}")
    
    if args.test_mode:
        print("Mode: TEST MODE (Publishing results only)")
    else:
        print("Mode: DATA COLLECTION")
    
    print("=" * 45)
    
    # Import and run the collector
    try:
        from yolo_dataset_collector import main as collector_main
        collector_main()
    except ImportError:
        print("Error: Cannot import yolo_dataset_collector.py")
        print("Make sure the file is in the same directory.")
        sys.exit(1)

if __name__ == '__main__':
    main() 