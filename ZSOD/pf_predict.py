#!/usr/bin/env python3
"""
YOLOE Prompt-Free Object Detection and Visualization
Based on: https://github.com/THU-MIG/yoloe

This script performs prompt-free object detection using YOLOE model
and visualizes the results with bounding boxes and labels.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLOE
import torch

def setup_model(model_name="yoloe-v8l-seg-pf.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Setup YOLOE model for prompt-free detection
    
    Args:
        model_name (str): Model checkpoint name
        device (str): Device to run inference on
    
    Returns:
        YOLOE: Loaded model instance
    """
    try:
        # Load pre-trained YOLOE model for prompt-free detection
        model = YOLOE.from_pretrained(f"jameslahm/{model_name.replace('.pt', '')}")
        model = model.to(device)
        print(f"✅ Model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Trying alternative loading method...")
        
        # Alternative loading method
        try:
            model = YOLOE(model_name)
            model = model.to(device)
            print(f"✅ Model loaded successfully on {device}")
            return model
        except Exception as e2:
            print(f"❌ Alternative loading failed: {e2}")
            raise

def predict_prompt_free(model, image_path, conf_threshold=0.3, iou_threshold=0.5):
    """
    Perform prompt-free object detection
    
    Args:
        model: YOLOE model instance
        image_path (str): Path to input image
        conf_threshold (float): Confidence threshold for detection
        iou_threshold (float): IoU threshold for NMS
    
    Returns:
        tuple: (results, original_image)
    """
    # Load image
    image = Image.open(image_path)
    
    # Perform prompt-free detection
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    return results, image

def visualize_detections(image, results, save_path=None, show_plot=True):
    """
    Visualize detection results with bounding boxes and labels
    
    Args:
        image: Original PIL Image
        results: YOLOE detection results
        save_path (str): Path to save annotated image
        show_plot (bool): Whether to display the plot
    
    Returns:
        np.ndarray: Annotated image array
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Process detection results
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        # Extract detection data
        xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
        conf = boxes.conf.cpu().numpy()  # Confidence scores
        cls = boxes.cls.cpu().numpy().astype(int)  # Class indices
        
        # Get class names (YOLOE uses a built-in vocabulary for prompt-free detection)
        names = results[0].names
        
        print(f"🔍 Detected {len(xyxy)} objects:")
        
        # Define colors for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(cls))))
        
        for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Get class name and color
            class_name = names.get(class_id, f"Class_{class_id}")
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with confidence
            label = f"{class_name}: {confidence:.2f}"
            ax.text(
                x1, y1 - 5, label,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='black', weight='bold'
            )
            
            print(f"  {i+1}. {class_name}: {confidence:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    else:
        print("🚫 No objects detected")
    
    ax.set_title("YOLOE Prompt-Free Object Detection", fontsize=16, weight='bold')
    ax.axis('off')
    
    # Save annotated image
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"💾 Annotated image saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    # Return annotated image as numpy array
    fig.canvas.draw()
    annotated_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_img = annotated_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return annotated_img

def main():
    """
    Main function to run prompt-free object detection
    """
    # Configuration
    IMAGE_PATH = "sample.png"  # Input image path
    MODEL_NAME = "yoloe-v8l-seg-pf.pt"  # Prompt-free model
    CONF_THRESHOLD = 0.3  # Confidence threshold (lower for more detections)
    IOU_THRESHOLD = 0.5   # IoU threshold for NMS
    SAVE_PATH = "sample_detected.png"  # Output image path
    
    print("🚀 Starting YOLOE Prompt-Free Object Detection")
    print("=" * 50)
    
    try:
        # Setup model
        print("📥 Loading YOLOE model...")
        model = setup_model(MODEL_NAME)
        
        # Perform detection
        print(f"🔍 Detecting objects in: {IMAGE_PATH}")
        results, original_image = predict_prompt_free(
            model, IMAGE_PATH, CONF_THRESHOLD, IOU_THRESHOLD
        )
        
        # Visualize results
        print("🎨 Visualizing detection results...")
        visualize_detections(
            original_image, results, 
            save_path=SAVE_PATH, show_plot=True
        )
        
        print("✅ Detection completed successfully!")
        
    except FileNotFoundError:
        print(f"❌ Image file not found: {IMAGE_PATH}")
        print("💡 Make sure 'sample.png' exists in the current directory")
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        print("💡 Troubleshooting tips:")
        print("   - Check if the image file exists")
        print("   - Ensure YOLOE is properly installed")
        print("   - Try lowering confidence threshold")

if __name__ == "__main__":
    main()