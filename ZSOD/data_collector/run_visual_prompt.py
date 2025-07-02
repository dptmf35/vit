#!/usr/bin/env python3

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Visual Prompt Detector Configuration')
    
    # Detection parameters
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--iou_threshold', type=float, default=0.4,
                       help='IOU threshold for NMS (default: 0.4)')
    
    # Input parameters
    parser.add_argument('--image_topic', type=str, default='/stereo_image_color',
                       help='Input image topic (default: /stereo_image_color)')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='yoloe-11m-seg.pt',
                       help='YOLOE model path (default: yoloe-11m-seg.pt)')
    
    args = parser.parse_args()
    
    # Set environment variables for the detector
    os.environ['VISUAL_CONF_THRESHOLD'] = str(args.conf_threshold)
    os.environ['VISUAL_IOU_THRESHOLD'] = str(args.iou_threshold)
    os.environ['VISUAL_IMAGE_TOPIC'] = args.image_topic
    os.environ['VISUAL_PROMPT_MODEL'] = args.model_path
    
    print("=== YOLOE Visual Prompt Detector ===")
    print("Official YOLOE visual prompting implementation")
    print(f"Confidence Threshold: {args.conf_threshold}")
    print(f"IOU Threshold: {args.iou_threshold}")
    print(f"Image Topic: {args.image_topic}")
    print(f"Model Path: {args.model_path}")
    print("=" * 45)
    print("YOLOE Visual Prompting Features:")
    print("  ✓ SAVPE (Semantic-Activated Visual Prompt Encoder)")
    print("  ✓ Official YOLOEVPSegPredictor support")
    print("  ✓ Fallback to prompt-guided detection")
    print("=" * 45)
    print("Controls:")
    print("  Left click + drag: Add box prompt (REQUIRED for YOLOE)")
    print("  Double click: Add point prompt (fallback method)")
    print("  'c': Clear all prompts")
    print("  'd': Run YOLOE visual prompt detection")
    print("  't': Test regular detection")
    print("  's': Save current detection results")
    print("  'q': Quit")
    print("=" * 45)
    print("Note: Box prompts are required for optimal YOLOE performance")
    
    # Import and run the detector
    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from visual_prompt_detector import main as detector_main
        detector_main()
    except ImportError as e:
        print(f"Error: Cannot import visual_prompt_detector.py")
        print(f"Import error: {e}")
        print("Make sure the file is in the same directory.")
        sys.exit(1)

if __name__ == '__main__':
    main() 