#!/usr/bin/env python3

import sys
import os
from pathlib import Path

def main():
    """Test script for visual annotation editor"""
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    print("=== Visual Annotation Editor Test ===")
    print()
    
    # Check if we have images in the dataset
    dataset_path = Path("~/yolo_dataset").expanduser()
    images_dir = dataset_path / "images"
    
    if not images_dir.exists():
        print("❌ No dataset directory found")
        print(f"Expected: {images_dir}")
        print()
        print("Please run the dataset collector first to create some images:")
        print("  python3 run_dataset_collector.py --test_mode --conf_threshold 0.4")
        print("  # Then switch to collection mode with 'c' key")
        return
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(ext)))
    
    if not image_files:
        print("❌ No images found in dataset")
        print(f"Dataset directory: {dataset_path}")
        print()
        print("Please collect some images first:")
        print("  python3 run_dataset_collector.py --conf_threshold 0.4")
        return
    
    print(f"✅ Found {len(image_files)} images")
    print()
    
    # Show available images
    print("Available images:")
    for i, img_path in enumerate(image_files[:10]):  # Show first 10
        print(f"  {i}: {img_path.name}")
    
    if len(image_files) > 10:
        print(f"  ... and {len(image_files) - 10} more")
    
    print()
    
    # Ask user to select an image
    try:
        if len(sys.argv) > 1:
            # Image index provided as argument
            choice = int(sys.argv[1])
        else:
            choice = int(input(f"Select image to edit (0-{len(image_files)-1}): "))
        
        if not (0 <= choice < len(image_files)):
            print(f"Invalid choice. Please select 0-{len(image_files)-1}")
            return
        
        selected_image = image_files[choice]
        
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or cancelled")
        return
    
    # Determine label path
    labels_dir = dataset_path / "labels"
    label_path = labels_dir / f"{selected_image.stem}.txt"
    
    print(f"Selected image: {selected_image}")
    print(f"Label file: {label_path}")
    print()
    
    # Import and run the editor
    try:
        from interactive_annotation_editor import InteractiveAnnotationEditor
        
        # Default class names (same as in collector)
            class_names = ["bed", "cabinet", "carpet", "chair", "closet", "countertop", "curtain", "desk", "door", "fridge",
                   "gas stove", "hanger", "lamp", "microwave", "nightstand", "plant", "shelf", "sofa", 
                   "table", "tv", "window", "vanity"]
        
        print("🎯 Starting Visual Annotation Editor...")
        print("Use mouse to draw bounding boxes, right-click to edit/delete")
        print()
        
        editor = InteractiveAnnotationEditor(str(selected_image), str(label_path), class_names)
        editor.run()
        
        print("✅ Visual editor closed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure interactive_annotation_editor.py is in the same directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main() 