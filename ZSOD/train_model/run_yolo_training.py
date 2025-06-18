#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='YOLO Training Launcher')
    
    # Dataset parameters
    parser.add_argument('--dataset-dir', type=str, default='~/yolo_dataset',
                       help='Dataset directory path (default: ~/yolo_dataset)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                       choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                       help='YOLO model size (default: yolo11s.pt)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto-determined if not set)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (default: 640)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='training_output',
                       help='Output directory for results (default: training_output)')
    
    # Analysis options
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze dataset without training')
    parser.add_argument('--quick-train', action='store_true',
                       help='Quick training with reduced epochs for testing')
    
    args = parser.parse_args()
    
    # Quick training mode
    if args.quick_train:
        args.epochs = 20
        print("🚀 Quick training mode: 20 epochs")
    
    print("=== YOLO Training Configuration ===")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size or 'auto'}")
    print(f"Image size: {args.imgsz}")
    print(f"Output: {args.output_dir}")
    print("=" * 35)
    
    # Set environment variables
    os.environ['YOLO_DATASET_DIR'] = args.dataset_dir
    os.environ['YOLO_MODEL'] = args.model
    os.environ['YOLO_EPOCHS'] = str(args.epochs)
    os.environ['YOLO_OUTPUT_DIR'] = args.output_dir
    
    if args.batch_size:
        os.environ['YOLO_BATCH_SIZE'] = str(args.batch_size)
    
    # Import and run trainer
    try:
        from train_yolo import main as train_main
        
        # Prepare arguments for train_yolo.py
        sys.argv = ['train_yolo.py',
                   '--dataset', args.dataset_dir,
                   '--model', args.model,
                   '--epochs', str(args.epochs),
                   '--imgsz', str(args.imgsz),
                   '--output-dir', args.output_dir]
        
        if args.batch_size:
            sys.argv.extend(['--batch-size', str(args.batch_size)])
        
        if args.analyze_only:
            sys.argv.append('--analyze-only')
        
        train_main()
        
    except ImportError:
        print("❌ Error: Cannot import train_yolo.py")
        print("Make sure train_yolo.py is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 