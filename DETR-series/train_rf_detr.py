import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers import TrainingArguments, Trainer
import json
from PIL import Image
import numpy as np

# --- Configuration ---
DATASET_PATH = "/home/yeseul/yolo_dataset_1015"
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
TRAIN_JSON = os.path.join(DATASET_PATH, "coco_annotations", "train.json")
VAL_JSON = os.path.join(DATASET_PATH, "coco_annotations", "valid.json")

# RF-DETR or RT-DETR checkpoint (Using RT-DETR-r18 as base, similar to RF-DETR usage)
MODEL_CHECKPOINT = "PekingU/rtdetr_r18vd" 

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_file, image_processor):
        self.img_dir = img_dir
        self.image_processor = image_processor
        with open(annotation_file) as f:
            self.coco = json.load(f)
        
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = Image.open(img_path).convert("RGB")
        
        # Prepare annotations
        boxes = []
        class_labels = []
        area = []
        iscrowd = []
        
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h]) # xywh -> xyxy
                class_labels.append(ann['category_id'])
                area.append(ann['area'])
                iscrowd.append(ann['iscrowd'])
        
        # No annotations case
        if len(boxes) == 0:
            boxes = np.zeros((0, 4))
            class_labels = np.zeros((0,), dtype=np.int64)
            area = np.zeros((0,), dtype=np.float32)
            iscrowd = np.zeros((0,), dtype=np.int64)
            
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "area": torch.tensor(area, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([image.height, image.width])
        }
        
        # Apply processor
        # The processor expects a list of dictionaries for annotations if format is "coco_detection"
        # Each dictionary should correspond to one image and contain "image_id" and "annotations"
        
        # Construct annotations in the format expected by RTDetrImageProcessor
        formatted_annotations = {'image_id': img_id, 'annotations': []}
        
        if img_id in self.annotations:
             for ann in self.annotations[img_id]:
                formatted_annotations['annotations'].append(ann)
        
        # DEBUG: Print first item's formatted annotations to check correctness
        if idx == 0:
            print(f"\n[DEBUG] Image ID: {img_id}")
            print(f"[DEBUG] Formatted Annotations: {formatted_annotations}")
        
        # If no annotations, pass empty list inside the dict structure as required
        
        encoding = self.image_processor(
            images=image, 
            annotations=formatted_annotations, 
            return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0] # remove batch dim
        
        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {"pixel_values": torch.stack(pixel_values), "labels": labels}

def main():
    # 1. Load Processor
    image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    
    # 2. Create Datasets
    train_dataset = CocoDataset(IMAGES_DIR, TRAIN_JSON, image_processor)
    val_dataset = CocoDataset(IMAGES_DIR, VAL_JSON, image_processor)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 3. Load Model
    # Get number of classes from the json
    with open(TRAIN_JSON) as f:
        coco_data = json.load(f)
        # Ensure class IDs are contiguous if needed, or use max ID + 1
        # Here we assume 0-indexed classes as per conversion script
        num_classes = len(coco_data['categories']) 
        
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="rf_detr_results",
        per_device_train_batch_size=4,
        num_train_epochs=100,  # Increased to 100
        fp16=True, # Use GPU acceleration
        save_steps=500, # Save less frequently
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=image_processor,
    )

    print("Starting training...")
    trainer.train()
    
    print("Training complete. Saving model and state...")
    trainer.save_model("rf_detr_finetuned")
    trainer.save_state()  # Explicitly save trainer state (losses, logs)

if __name__ == "__main__":
    main()

