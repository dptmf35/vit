#!/usr/bin/env python3
"""
YOLOE Home Furniture Object Detection
Based on official YOLOE implementation with home-focused vocabulary

This script performs object detection using YOLOE with 80 classes
focused on home furniture and household items, similar to COCO dataset size.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLOE
import torch
import os

def get_home_furniture_classes():
    """
    Get 80 home furniture and household item classes
    Focused on items commonly found in home environments
    """
    home_classes = [
        # Living Room Furniture
        "sofa", "chair", "armchair", "coffee table", "tv stand", "bookshelf", "cabinet", "drawer",
        "cushion", "pillow", "blanket", "lamp", "floor lamp", "table lamp", "ceiling fan", "curtain",
        
        # Dining Room
        "dining table", "dining chair", "bar stool", "sideboard", "china cabinet", "wine rack",
        
        # Kitchen Items
        "refrigerator", "microwave", "oven", "toaster", "coffee maker", "blender", "kettle", "sink",
        "dishwasher", "kitchen cabinet", "kitchen counter", "cutting board", "knife", "fork", "spoon",
        "plate", "bowl", "cup", "glass", "bottle", "pot", "pan",
        
        # Bedroom Furniture
        "bed", "mattress", "nightstand", "dresser", "wardrobe", "mirror", "closet", "bedsheet",
        
        # Bathroom Items
        "toilet", "bathtub", "shower", "bathroom sink", "towel", "toilet paper", "soap", "toothbrush",
        
        # Electronics & Appliances
        "tv", "computer", "laptop", "tablet", "phone", "speaker", "headphones", "remote control",
        "keyboard", "mouse", "printer", "camera", "clock", "air conditioner", "heater", "vacuum",
        
        # Storage & Organization
        "box", "basket", "bag", "suitcase", "backpack", "handbag", "shelf", "hook",
        
        # Decorative Items
        "picture frame", "painting", "vase", "candle", "plant", "flower pot", "sculpture", "ornament",
        
        # Common Objects
        "book", "magazine", "newspaper", "pen", "pencil", "scissors", "tape", "key"
    ]
    
    # Ensure exactly 80 classes
    return home_classes[:80]

def setup_yoloe_home_model(model_size="v8s", device="cuda"):
    """
    Setup YOLOE model for home furniture detection
    
    Args:
        model_size: Model size (v8s, v8m, v8l)
        device: Device to run on
    """
    print(f"🏠 Setting up YOLOE-{model_size} for home furniture detection...")
    
    try:
        # Load YOLOE model
        model_name = f"jameslahm/yoloe-{model_size}-seg"
        print(f"📥 Loading {model_name}...")
        
        model = YOLOE.from_pretrained(model_name)
        
        if device == "cuda" and torch.cuda.is_available():
            model.cuda()
            print("✅ Model loaded on CUDA")
        else:
            model.cpu()
            print("✅ Model loaded on CPU")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        
        # Fallback to smaller model
        if model_size != "v8s":
            print("💡 Trying smaller v8s model...")
            return setup_yoloe_home_model("v8s", device)
        else:
            raise

def predict_home_furniture(model, image_path, home_classes, conf_threshold=0.25):
    """
    Perform home furniture detection
    
    Args:
        model: YOLOE model
        image_path: Path to image
        home_classes: List of home furniture classes
        conf_threshold: Confidence threshold
    """
    print(f"🔍 Detecting home furniture in: {image_path}")
    print(f"📝 Using {len(home_classes)} home-focused classes")
    print(f"🏷️ Classes: {home_classes[:10]}... (and {len(home_classes)-10} more)")
    
    # Load image
    image = Image.open(image_path)
    
    # Perform detection with text prompts
    results = model.predict(
        source=image,
        prompts=home_classes,
        conf=conf_threshold,
        iou=0.5,
        verbose=True,
        save=False
    )
    
    # Extract detected classes
    detected_classes = []
    if len(results) > 0 and results[0].boxes is not None:
        cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        for cls_idx in cls_indices:
            if cls_idx < len(home_classes):
                detected_classes.append(home_classes[cls_idx])
    
    detected_classes = list(set(detected_classes))  # Remove duplicates
    
    return results, image, detected_classes

def visualize_home_detection(image, results, home_classes, detected_classes, save_path=None):
    """
    Visualize home furniture detection results
    """
    img_array = np.array(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(img_array)
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        
        print(f"🏠 Detected {len(xyxy)} home items:")
        print(f"🎯 Found classes: {detected_classes}")
        
        # Use distinct colors for different furniture types
        colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20 distinct colors
        
        for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            if class_id < len(home_classes):
                class_name = home_classes[class_id]
            else:
                class_name = f"Unknown_{class_id}"
            
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with confidence
            label = f"{class_name}: {confidence:.2f}"
            ax.text(
                x1, y1 - 8, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=10, color='black', weight='bold'
            )
            
            print(f"  {i+1}. {class_name}: {confidence:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("🚫 No home furniture detected")
        print("💡 Try:")
        print("   - Lower confidence threshold")
        print("   - Different image with clear furniture")
        print("   - Check if image contains household items")
    
    ax.set_title("🏠 YOLOE Home Furniture Detection (80 Classes)", 
                fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"💾 Result saved: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close()

def print_detection_summary(detected_classes, home_classes):
    """
    Print detection summary by category
    """
    if not detected_classes:
        print("📊 No objects detected to summarize")
        return
    
    # Categorize detected items
    categories = {
        "Living Room": ["sofa", "chair", "armchair", "coffee table", "tv stand", "bookshelf", "cabinet", 
                       "cushion", "pillow", "blanket", "lamp", "floor lamp", "table lamp", "tv"],
        "Kitchen": ["refrigerator", "microwave", "oven", "toaster", "coffee maker", "blender", "kettle", 
                   "sink", "dishwasher", "plate", "bowl", "cup", "glass", "bottle", "pot", "pan"],
        "Bedroom": ["bed", "mattress", "nightstand", "dresser", "wardrobe", "mirror", "closet"],
        "Bathroom": ["toilet", "bathtub", "shower", "bathroom sink", "towel", "toilet paper", "soap"],
        "Electronics": ["tv", "computer", "laptop", "tablet", "phone", "speaker", "headphones", "remote control"],
        "Storage": ["box", "basket", "bag", "suitcase", "backpack", "handbag", "shelf"],
        "Decorative": ["picture frame", "painting", "vase", "candle", "plant", "flower pot"]
    }
    
    print("\n📊 Detection Summary by Category:")
    print("=" * 40)
    
    total_detected = 0
    for category, items in categories.items():
        found_items = [item for item in detected_classes if item in items]
        if found_items:
            print(f"🏷️ {category}: {', '.join(found_items)}")
            total_detected += len(found_items)
    
    # Items not in predefined categories
    uncategorized = [item for item in detected_classes 
                    if not any(item in items for items in categories.values())]
    if uncategorized:
        print(f"📦 Other: {', '.join(uncategorized)}")
        total_detected += len(uncategorized)
    
    print(f"\n✅ Total detected: {total_detected} items from {len(home_classes)} possible classes")
    print(f"📈 Detection rate: {total_detected/len(home_classes)*100:.1f}%")

def main():
    """
    Main function for home furniture detection
    """
    # Configuration
    IMAGE_PATH = "sample.png"  # Change to your image path
    MODEL_SIZE = "v8l"  # v8s, v8m, or v8l
    CONF_THRESHOLD = 0.1  # Lower threshold for home items
    SAVE_PATH = "home_furniture_detected.png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🏠 YOLOE Home Furniture Detection")
    print("=" * 50)
    print(f"📖 Detecting 80 home furniture & household items")
    print(f"🎯 Model: YOLOE-{MODEL_SIZE}")
    print(f"💾 Device: {DEVICE}")
    print("=" * 50)
    
    # Check VRAM
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {gpu_memory:.1f} GB")
        
        # Adjust model size based on VRAM
        if gpu_memory < 6 and MODEL_SIZE != "v8s":
            MODEL_SIZE = "v8s"
            print(f"⚠️ Switching to {MODEL_SIZE} due to limited VRAM")
    
    try:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get home furniture classes
        home_classes = get_home_furniture_classes()
        print(f"📝 Loaded {len(home_classes)} home furniture classes")
        
        # Setup model
        model = setup_yoloe_home_model(MODEL_SIZE, DEVICE)
        
        # Perform detection
        print(f"\n🔍 Analyzing image: {IMAGE_PATH}")
        results, image, detected_classes = predict_home_furniture(
            model, IMAGE_PATH, home_classes, CONF_THRESHOLD
        )
        
        # Visualize results
        print("\n🎨 Creating visualization...")
        visualize_home_detection(image, results, home_classes, detected_classes, SAVE_PATH)
        
        # Print summary
        print_detection_summary(detected_classes, home_classes)
        
        print("\n✅ Home furniture detection completed!")
        if detected_classes:
            print(f"🎉 Successfully detected: {', '.join(detected_classes[:10])}")
            if len(detected_classes) > 10:
                print(f"    ... and {len(detected_classes)-10} more items")
        else:
            print("🤔 No furniture detected. Try:")
            print("   - Lower confidence threshold (current: {:.2f})".format(CONF_THRESHOLD))
            print("   - Image with clear household items")
            print("   - Indoor scene with furniture")
        
    except FileNotFoundError:
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("💡 Please check the image path and ensure the file exists")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned up")

if __name__ == "__main__":
    main()