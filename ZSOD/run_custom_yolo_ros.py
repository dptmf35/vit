#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch Custom YOLO ROS2 Detector')
    
    # Model options
    parser.add_argument('--model', type=str, default='train_model/training_output/train/weights/best.pt',
                       help='Path to trained YOLO model (default: best.pt from training)')
    
    # Camera options
    parser.add_argument('--camera-topic', type=str, default='/stereo_image_color',
                       help='Camera topic to subscribe to (default: /stereo_image_color)')
    parser.add_argument('--usb-camera', action='store_true',
                       help='Use USB camera (topic: /usb_cam/image_raw)')
    parser.add_argument('--realsense', action='store_true',
                       help='Use RealSense camera (topic: /camera/color/image_raw)')
    
    # Detection options
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--no-annotated', action='store_true',
                       help='Disable annotated image publishing')
    parser.add_argument('--save-detections', action='store_true',
                       help='Save detection results to files')
    
    # Model selection
    parser.add_argument('--use-last', action='store_true',
                       help='Use last.pt instead of best.pt')
    
    args = parser.parse_args()
    
    # Set camera topic based on camera type
    if args.usb_camera:
        camera_topic = '/usb_cam/image_raw'
    elif args.realsense:
        camera_topic = '/camera/color/image_raw'
    else:
        camera_topic = args.camera_topic
    
    # Set model path
    if args.use_last:
        model_path = args.model.replace('best.pt', 'last.pt')
    else:
        model_path = args.model
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("\nAvailable models:")
        weights_dir = Path("train_model/training_output/train/weights")
        if weights_dir.exists():
            for pt_file in weights_dir.glob("*.pt"):
                print(f"  {pt_file}")
        else:
            print("  No trained models found. Please train a model first.")
        sys.exit(1)
    
    print("🚀 Launching Custom YOLO Detector")
    print("=" * 40)
    print(f"Model: {model_path}")
    print(f"Camera topic: {camera_topic}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Publish annotated images: {not args.no_annotated}")
    print(f"Save detections: {args.save_detections}")
    print("=" * 40)
    
    # Build ROS2 command
    cmd = [
        'python3', 'ros_custom_yolo_detector.py',
        '--ros-args',
        '-p', f'model_path:={model_path}',
        '-p', f'camera_topic:={camera_topic}',
        '-p', f'confidence_threshold:={args.confidence}',
        '-p', f'publish_annotated:={not args.no_annotated}',
        '-p', f'save_detections:={args.save_detections}'
    ]
    
    try:
        print("📡 Starting ROS2 node...")
        print("📹 Topics will be published to:")
        print(f"   - /custom_yolo/detections")
        print(f"   - /custom_yolo/bounding_boxes")
        if not args.no_annotated:
            print(f"   - /custom_yolo/annotated_image")
        print(f"\n📱 Monitor topics:")
        print(f"   ros2 topic echo /custom_yolo/detections")
        print(f"   ros2 run rqt_image_view rqt_image_view /custom_yolo/annotated_image")
        print(f"\n🛑 Press Ctrl+C to stop\n")
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running detector: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Python3 not found or ROS2 not sourced")
        print("Make sure to source ROS2: source /opt/ros/humble/setup.bash")
        sys.exit(1)

if __name__ == '__main__':
    main() 