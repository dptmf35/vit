#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='YOLO Dataset Reviewer - Launcher Script')
    parser.add_argument('--dataset-dir', type=str, default='../yolo_dataset', 
                       help='Dataset directory path (default: ../yolo_dataset)')
    parser.add_argument('--start-index', type=int, default=0, 
                       help='Starting image index (default: 0)')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        print("Please check the path or create a dataset first.")
        return 1
    
    # Check for required directories
    required_dirs = ['images', 'labels']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (dataset_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"Error: Missing required directories: {missing_dirs}")
        print(f"Expected structure:")
        print(f"  {dataset_dir}/")
        print(f"    ├── images/")
        print(f"    ├── labels/")
        print(f"    └── visualizations/ (optional)")
        return 1
    
    # Import and run reviewer
    from dataset_reviewer import DatasetReviewer
    
    print(f"Starting YOLO Dataset Reviewer...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Starting index: {args.start_index}")
    print("-" * 50)
    
    try:
        reviewer = DatasetReviewer(dataset_dir)
        reviewer.current_index = args.start_index
        reviewer.run()
    except Exception as e:
        print(f"Error running reviewer: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 