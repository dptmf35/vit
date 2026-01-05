#!/usr/bin/env python3

import sys
import os
import threading
from pathlib import Path
from typing import Optional, Callable, Dict
import yaml
import shutil

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "train_model"))

from config.pipeline_config import TrainingConfig

class TrainingModule:
    """Module for training YOLO models"""

    def __init__(self, config: TrainingConfig, dataset_path: str, status_callback: Optional[Callable] = None):
        """
        Initialize training module

        Args:
            config: Training configuration
            dataset_path: Path to dataset directory
            status_callback: Optional callback function for status updates
        """
        self.config = config
        self.dataset_path = Path(dataset_path).expanduser()
        self.status_callback = status_callback
        self.is_training = False
        self.trainer = None

        # Output directory
        self.output_dir = Path(self.config.output_dir).expanduser()

    def _update_status(self, message: str, level: str = 'info'):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(message, level)

    def validate_dataset(self) -> bool:
        """Validate dataset before training"""
        if not self.dataset_path.exists():
            self._update_status(f"Dataset not found: {self.dataset_path}", 'error')
            return False

        # Check required directories
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            self._update_status("Dataset missing images or labels directory", 'error')
            return False

        # Check for images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(images_dir.glob(ext)))

        if not images:
            self._update_status("No images found in dataset", 'error')
            return False

        # Check for dataset.yaml
        yaml_path = self.dataset_path / "dataset.yaml"
        if not yaml_path.exists():
            self._update_status("dataset.yaml not found", 'error')
            return False

        self._update_status(f"Dataset validated: {len(images)} images", 'success')
        return True

    def analyze_dataset(self) -> Dict:
        """Analyze dataset composition"""
        analysis = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'avg_annotations_per_image': 0.0,
            'recommendations': []
        }

        if not self.dataset_path.exists():
            return analysis

        # Load class names
        class_names = self._load_class_names()

        # Analyze images and labels
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(images_dir.glob(ext)))

        analysis['total_images'] = len(images)

        for img_path in images:
            label_path = labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                class_id = int(line.split()[0])
                                class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                                analysis['class_distribution'][class_name] = analysis['class_distribution'].get(class_name, 0) + 1
                                analysis['total_annotations'] += 1
                            except:
                                continue

        if analysis['total_images'] > 0:
            analysis['avg_annotations_per_image'] = analysis['total_annotations'] / analysis['total_images']

        # Generate recommendations
        if analysis['total_images'] < 20:
            analysis['recommendations'].append("Small dataset (<20 images): Use transfer learning and lower learning rate")

        if analysis['total_annotations'] < 50:
            analysis['recommendations'].append("Few annotations (<50 objects): Each class needs 10+ examples ideally")

        rare_classes = [name for name, count in analysis['class_distribution'].items() if count < 5]
        if rare_classes:
            analysis['recommendations'].append(f"Classes with <5 examples: {', '.join(rare_classes[:5])}")

        return analysis

    def _load_class_names(self):
        """Load class names from dataset.yaml"""
        yaml_path = self.dataset_path / "dataset.yaml"

        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        names = data['names']
                        if isinstance(names, dict):
                            return [names[i] for i in sorted(names)]
                        elif isinstance(names, list):
                            return names
            except:
                pass

        return []

    def start_training(self, background: bool = False):
        """
        Start model training

        Args:
            background: If True, run training in background thread
        """
        if self.is_training:
            self._update_status("Training is already running", 'warning')
            return False

        if not self.validate_dataset():
            return False

        # Analyze dataset first
        self._update_status("Analyzing dataset...", 'info')
        analysis = self.analyze_dataset()

        self._update_status(f"Dataset: {analysis['total_images']} images, {analysis['total_annotations']} annotations", 'info')

        if analysis['recommendations']:
            for rec in analysis['recommendations']:
                self._update_status(f"Recommendation: {rec}", 'warning')

        if background:
            # Run in background thread
            self.training_thread = threading.Thread(target=self._train, daemon=False)
            self.training_thread.start()
            return True
        else:
            # Run in foreground
            return self._train()

    def _train(self):
        """Internal training method"""
        try:
            from train_yolo import YOLOTrainer

            self.is_training = True
            self._update_status("Starting training...", 'info')

            # Create trainer
            self.trainer = YOLOTrainer(str(self.dataset_path), str(self.output_dir))

            # Train
            best_model, results = self.trainer.train(
                model_name=self.config.model_size,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                imgsz=self.config.image_size
            )

            self.is_training = False

            if best_model:
                self._update_status(f"Training completed! Model: {best_model}", 'success')
                return True
            else:
                self._update_status("Training failed", 'error')
                return False

        except Exception as e:
            self.is_training = False
            self._update_status(f"Training error: {e}", 'error')
            return False

    def get_best_model_path(self) -> Optional[Path]:
        """Get path to best trained model"""
        best_path = self.output_dir / 'train' / 'weights' / 'best.pt'

        if best_path.exists():
            return best_path

        last_path = self.output_dir / 'train' / 'weights' / 'last.pt'
        if last_path.exists():
            return last_path

        return None

    def get_training_results(self) -> Dict:
        """Get training results and metrics"""
        results = {
            'model_path': None,
            'training_completed': False,
            'metrics': {}
        }

        model_path = self.get_best_model_path()
        if model_path:
            results['model_path'] = str(model_path)
            results['training_completed'] = True

            # Try to load results
            results_csv = self.output_dir / 'train' / 'results.csv'
            if results_csv.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(results_csv)

                    # Get final metrics
                    if not df.empty:
                        last_row = df.iloc[-1]
                        results['metrics'] = {
                            'final_epoch': int(last_row.get('epoch', 0)),
                            'train_loss': float(last_row.get('train/box_loss', 0)),
                            'val_loss': float(last_row.get('val/box_loss', 0)),
                            'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                            'mAP50-95': float(last_row.get('metrics/mAP50-95(B)', 0))
                        }
                except:
                    pass

        return results

# Example usage
if __name__ == '__main__':
    from config.pipeline_config import TrainingConfig

    # Status callback
    def status_callback(message, level):
        print(f"[{level.upper()}] {message}")

    # Create configuration
    config = TrainingConfig(
        model_size='yolo11s.pt',
        epochs=50,
        batch_size=4
    )

    # Create module
    module = TrainingModule(config, "~/yolo_dataset", status_callback)

    # Analyze dataset
    print("\nDataset Analysis:")
    analysis = module.analyze_dataset()
    print(f"Total images: {analysis['total_images']}")
    print(f"Total annotations: {analysis['total_annotations']}")
    print(f"Avg annotations/image: {analysis['avg_annotations_per_image']:.1f}")

    print("\nClass distribution:")
    for class_name, count in sorted(analysis['class_distribution'].items()):
        print(f"  {class_name}: {count}")

    if analysis['recommendations']:
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
