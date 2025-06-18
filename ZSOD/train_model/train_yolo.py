#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOTrainer:
    def __init__(self, dataset_path, output_dir="training_output"):
        # Expand user path (~) properly
        self.dataset_path = Path(dataset_path).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(exist_ok=True)
        
        # Check dataset structure
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError(f"Dataset structure invalid. Expected {self.images_dir} and {self.labels_dir}")
        
        # Load dataset info
        self.class_names, self.num_classes = self.load_class_info()
        self.image_files = self.get_image_files()
        
        print(f"Dataset: {self.dataset_path}")
        print(f"Classes ({self.num_classes}): {self.class_names}")
        print(f"Images: {len(self.image_files)}")
        
    def load_class_info(self):
        """Load class information from dataset.yaml or use defaults"""
        yaml_path = self.dataset_path / "dataset.yaml"
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    class_names = data['names']
                    if isinstance(class_names, list):
                        return class_names, len(class_names)
        
        # Default classes
        class_names = ["bed", "cabinet", "carpet", "chair", "closet", "curtain", "desk", "door", "fridge",
                      "gas stove", "hanger", "lamp", "microwave", "nightstand", "plant", "shelf", "sofa", 
                      "table", "tv", "window", "vanity"]
        
        return class_names, len(class_names)
    
    def get_image_files(self):
        """Get all image files with corresponding labels"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in extensions:
            for img_path in self.images_dir.glob(ext):
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    image_files.append(img_path)
        
        return sorted(image_files)
    
    def create_train_val_split(self, val_ratio=0.2):
        """Create train/validation split optimized for small datasets"""
        
        if len(self.image_files) < 5:
            print("⚠️  Very small dataset. Using all data for training.")
            return self.image_files, []
        
        # For small datasets, ensure at least 2 validation samples
        num_val = max(2, int(len(self.image_files) * val_ratio))
        num_val = min(num_val, len(self.image_files) - 3)  # Keep at least 3 for training
        
        # Simple random split
        import random
        random.seed(42)
        shuffled = self.image_files.copy()
        random.shuffle(shuffled)
        
        val_imgs = shuffled[:num_val]
        train_imgs = shuffled[num_val:]
        
        print(f"Train/Val split: {len(train_imgs)}/{len(val_imgs)}")
        return train_imgs, val_imgs
    
    def create_yolo_dataset(self, train_imgs, val_imgs):
        """Create YOLO format dataset structure"""
        
        # Create directories
        train_images_dir = self.output_dir / "train" / "images"
        train_labels_dir = self.output_dir / "train" / "labels"
        val_images_dir = self.output_dir / "val" / "images"
        val_labels_dir = self.output_dir / "val" / "labels"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy training files
        print("Copying training files...")
        for img_path in train_imgs:
            shutil.copy2(img_path, train_images_dir / img_path.name)
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, train_labels_dir / f"{img_path.stem}.txt")
        
        # Copy validation files
        if val_imgs:
            print("Copying validation files...")
            for img_path in val_imgs:
                shutil.copy2(img_path, val_images_dir / img_path.name)
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, val_labels_dir / f"{img_path.stem}.txt")
        
        # Create dataset.yaml
        dataset_yaml = {
            'train': str(train_images_dir.absolute()),
            'val': str(val_images_dir.absolute()) if val_imgs else str(train_images_dir.absolute()),
            'nc': self.num_classes,
            'names': self.class_names
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"Created dataset.yaml: {yaml_path}")
        return yaml_path
    
    def train(self, model_name="yolo11s.pt", epochs=100, imgsz=640, batch_size=None):
        """Train YOLO model with small dataset optimizations"""
        
        # Auto-adjust parameters for small datasets
        if batch_size is None:
            if len(self.image_files) < 10:
                batch_size = 2
            elif len(self.image_files) < 30:
                batch_size = 4
            else:
                batch_size = 8
        
        # Adjust training settings for small datasets
        if len(self.image_files) < 20:
            epochs = min(epochs, 200)
            patience = 30
        else:
            patience = 50
        
        print(f"\n=== Training Configuration ===")
        print(f"Model: {model_name}")
        print(f"Images: {len(self.image_files)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {imgsz}")
        print("=" * 30)
        
        # Create dataset split
        train_imgs, val_imgs = self.create_train_val_split()
        dataset_yaml_path = self.create_yolo_dataset(train_imgs, val_imgs)
        
        # Initialize model
        model = YOLO(model_name)
        
        # Training arguments for small datasets
        train_args = {
            'data': str(dataset_yaml_path),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'patience': patience,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            
            # Optimized for small datasets
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            
            # Conservative data augmentation
            'hsv_h': 0.01,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.2,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.3,
            'mixup': 0.0,
            
            'val': True,
            'plots': True,
            'save_json': True,
            'verbose': True
        }
        
        print("\n🚀 Starting training...")
        
        try:
            results = model.train(**train_args)
            
            print("\n✅ Training completed!")
            
            # Find best model
            best_model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
            if best_model_path.exists():
                print(f"Best model: {best_model_path}")
                return best_model_path, results
            else:
                last_model_path = self.output_dir / 'train' / 'weights' / 'last.pt'
                print(f"Last model: {last_model_path}")
                return last_model_path, results
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None, None
    
    def analyze_dataset(self):
        """Analyze dataset composition"""
        print("\n=== Dataset Analysis ===")
        
        class_counts = {}
        total_annotations = 0
        
        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                class_id = int(line.split()[0])
                                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                                total_annotations += 1
                            except:
                                continue
        
        print(f"Total images: {len(self.image_files)}")
        print(f"Total annotations: {total_annotations}")
        print(f"Avg annotations/image: {total_annotations/len(self.image_files):.1f}")
        
        print("\nClass distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        
        # Recommendations
        print("\n=== Recommendations ===")
        if len(self.image_files) < 20:
            print("⚠️  Small dataset (<20 images)")
            print("  - Use transfer learning (pre-trained weights)")
            print("  - Lower learning rate")
            print("  - More training epochs")
            
        if total_annotations < 50:
            print("⚠️  Few annotations (<50 objects)")
            print("  - Each class needs 10+ examples ideally")
            
        rare_classes = [name for name, count in class_counts.items() if count < 5]
        if rare_classes:
            print(f"⚠️  Classes with <5 examples: {', '.join(rare_classes)}")

def main():
    parser = argparse.ArgumentParser(description='Train YOLO on small dataset')
    parser.add_argument('--dataset', type=str, default='~/yolo_dataset',
                       help='Dataset directory path')
    parser.add_argument('--model', type=str, default='yolo11s.pt',
                       help='YOLO model (yolo11n.pt, yolo11s.pt, yolo11m.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (auto if not set)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--output-dir', type=str, default='training_output',
                       help='Output directory')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze dataset')
    
    args = parser.parse_args()
    
    try:
        trainer = YOLOTrainer(args.dataset, args.output_dir)
        trainer.analyze_dataset()
        
        if args.analyze_only:
            return
        
        print(f"\nStart training? Dataset: {len(trainer.image_files)} images")
        response = input("Continue? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
        
        # Train
        best_model, results = trainer.train(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz
        )
        
        if best_model:
            print(f"\n🎉 Success! Model: {best_model}")
            print("\nUsage:")
            print(f"  model = YOLO('{best_model}')")
            print("  results = model.predict('image.jpg')")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main() 