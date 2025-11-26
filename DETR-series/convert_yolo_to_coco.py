import os
import json
import yaml
import glob
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def convert_yolo_to_coco(image_paths, label_dir, classes, output_file):
    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(classes)]
    
    annotation_id = 0
    
    print(f"Converting {len(image_paths)} images to {output_file}...")
    
    for img_id, img_path in enumerate(tqdm(image_paths)):
        img_path = Path(img_path)
        # Image info
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue
            
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "height": height,
            "width": width
        })
        
        # Label info
        label_file = Path(label_dir) / f"{img_path.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Convert YOLO (normalized center) to COCO (absolute top-left xywh)
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                abs_w = w * width
                abs_h = h * height
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
    print(f"Saved {output_file}")

def main():
    # Configuration
    dataset_path = "/home/yeseul/yolo_dataset_1015"
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")
    
    # Output directory for COCO json
    output_dir = os.path.join(dataset_path, "coco_annotations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load classes
    data_yaml = load_yaml(yaml_path)
    classes = data_yaml['names']
    
    # Get all images
    all_images = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                 glob.glob(os.path.join(image_dir, "*.png")) + \
                 glob.glob(os.path.join(image_dir, "*.jpeg"))
    
    if not all_images:
        print("No images found!")
        return

    # Split train/val (80/20 split)
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    # Convert
    convert_yolo_to_coco(train_imgs, label_dir, classes, os.path.join(output_dir, "train.json"))
    convert_yolo_to_coco(val_imgs, label_dir, classes, os.path.join(output_dir, "valid.json"))
    
    # Generate class map file (often used in Roboflow notebooks)
    with open(os.path.join(output_dir, "class_map.json"), "w") as f:
        json.dump({str(i): name for i, name in enumerate(classes)}, f, indent=4)

if __name__ == "__main__":
    main()

