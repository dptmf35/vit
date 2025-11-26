import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import random

# --- Configuration ---
DATASET_PATH = "/home/yeseul/yolo_dataset_1015"
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
VAL_JSON = os.path.join(DATASET_PATH, "coco_annotations", "valid.json")
CLASS_MAP_FILE = os.path.join(DATASET_PATH, "coco_annotations", "class_map.json")
# Use absolute path to ensure the model is found
MODEL_PATH = "/home/yeseul/Desktop/mygitrepos/vit/DETR-series/rf_detr_finetuned"  
OUTPUT_DIR = "/home/yeseul/Desktop/mygitrepos/vit/DETR-series/evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_class_map():
    with open(CLASS_MAP_FILE, 'r') as f:
        return json.load(f)

def plot_loss_curve(trainer_state_path):
    """Plots training loss from trainer_state.json"""
    if not os.path.exists(trainer_state_path):
        print(f"No trainer_state.json found at {trainer_state_path}")
        return

    with open(trainer_state_path, 'r') as f:
        data = json.load(f)
    
    log_history = data.get('log_history', [])
    
    steps = []
    losses = []
    epochs = []
    
    for entry in log_history:
        if 'loss' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            epochs.append(entry['epoch'])
            
    if not steps:
        print("No loss data found in logs.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    print(f"Loss curve saved to {os.path.join(OUTPUT_DIR, 'loss_curve.png')}")

def visualize_predictions(model, processor, image_paths, class_map, threshold=0.5, num_samples=5):
    """Runs inference and saves visualized images"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    samples = random.sample(image_paths, min(len(image_paths), num_samples))
    
    print(f"Visualizing {len(samples)} predictions...")
    
    for i, img_path in enumerate(samples):
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
        draw = ImageDraw.Draw(image)
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_str = class_map.get(str(label.item()), str(label.item()))
            
            # Draw box
            draw.rectangle(box, outline="red", width=3)
            
            # Draw label
            text = f"{label_str}: {round(score.item(), 3)}"
            # Simple text drawing (might need a font for better look, but default is okay)
            draw.text((box[0], box[1]), text, fill="red")

        save_path = os.path.join(OUTPUT_DIR, f"pred_{os.path.basename(img_path)}")
        image.save(save_path)
        print(f"Saved prediction: {save_path}")

def main():
    print("Loading model...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForObjectDetection.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        print("Make sure you have run the training script first.")
        return

    # 1. Plot Loss Curve
    # Hugging Face Trainer saves training state in the output dir
    trainer_state_path = os.path.join(MODEL_PATH, "trainer_state.json")
    plot_loss_curve(trainer_state_path)

    # 2. Visualize Predictions
    class_map = load_class_map()
    
    # Get validation images
    with open(VAL_JSON, 'r') as f:
        val_data = json.load(f)
        val_images = [os.path.join(IMAGES_DIR, img['file_name']) for img in val_data['images']]
    
    visualize_predictions(model, processor, val_images, class_map, threshold=0.05, num_samples=10)
    
    # Note: For full mAP calculation (COCO Eval), we would need to run inference on the whole val set
    # and use pycocotools. That requires more setup. 
    # For now, visual inspection and loss curve are good first steps.
    
    print("\nEvaluation complete!")
    print(f"Check the '{OUTPUT_DIR}' directory for results.")

if __name__ == "__main__":
    main()

