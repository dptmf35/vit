#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.pipeline_config import CollectionConfig

class AnnotationReviewModule:
    """Module for reviewing and editing dataset annotations"""

    def __init__(self, dataset_path: str, status_callback: Optional[Callable] = None):
        """
        Initialize annotation review module

        Args:
            dataset_path: Path to dataset directory
            status_callback: Optional callback function for status updates
        """
        self.dataset_path = Path(dataset_path).expanduser()
        self.status_callback = status_callback

        # Get paths relative to data_collector directory
        self.data_collector_dir = Path(__file__).parent.parent.parent
        self.reviewer_script = self.data_collector_dir / "run_dataset_reviewer.py"

    def _update_status(self, message: str, level: str = 'info'):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(message, level)

    def validate_dataset(self) -> bool:
        """Validate dataset structure before review"""
        if not self.dataset_path.exists():
            self._update_status(f"Dataset not found: {self.dataset_path}", 'error')
            return False

        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                self._update_status(f"Required directory missing: {dir_name}", 'error')
                return False

        # Check if there are images
        images_dir = self.dataset_path / "images"
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(images_dir.glob(ext)))

        if not images:
            self._update_status("No images found in dataset", 'error')
            return False

        self._update_status(f"Dataset valid: {len(images)} images found", 'success')
        return True

    def launch_reviewer(self, start_index: int = 0):
        """
        Launch the annotation reviewer GUI

        Args:
            start_index: Image index to start from
        """
        if not self.validate_dataset():
            return False

        try:
            self._update_status("Launching annotation reviewer...", 'info')

            # Prepare command
            cmd = [
                'python3',
                str(self.reviewer_script),
                '--dataset-dir', str(self.dataset_path),
                '--start-index', str(start_index)
            ]

            # Launch reviewer as separate process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            self._update_status("Annotation reviewer launched", 'success')
            self._update_status("Close the reviewer window when finished", 'info')

            # Wait for process to complete
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self._update_status("Annotation review completed", 'success')
                return True
            else:
                self._update_status(f"Reviewer error: {stderr}", 'error')
                return False

        except Exception as e:
            self._update_status(f"Failed to launch reviewer: {e}", 'error')
            return False

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            'total_images': 0,
            'total_labels': 0,
            'images_with_labels': 0,
            'images_without_labels': 0,
            'class_distribution': {}
        }

        if not self.dataset_path.exists():
            return stats

        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"

        # Count images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(images_dir.glob(ext)))

        stats['total_images'] = len(images)

        # Load class names
        class_names = self._load_class_names()

        # Analyze labels
        for img_path in images:
            label_path = labels_dir / f"{img_path.stem}.txt"

            if label_path.exists():
                stats['images_with_labels'] += 1

                # Parse label file
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

                                    stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
                                    stats['total_labels'] += 1
                            except:
                                continue
            else:
                stats['images_without_labels'] += 1

        return stats

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

        # Default class names
        return ["bed", "cabinet", "carpet", "chair", "closet", "curtain", "desk", "door",
                "fridge", "gas stove", "hanger", "lamp", "microwave", "nightstand",
                "plant", "shelf", "sofa", "table", "tv", "window", "vanity"]

    def validate_labels(self) -> Dict:
        """Validate all label files for format errors"""
        validation_results = {
            'valid_labels': 0,
            'invalid_labels': 0,
            'errors': []
        }

        if not self.dataset_path.exists():
            return validation_results

        labels_dir = self.dataset_path / "labels"
        if not labels_dir.exists():
            return validation_results

        label_files = list(labels_dir.glob('*.txt'))

        for label_path in label_files:
            try:
                with open(label_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) != 5:
                            validation_results['invalid_labels'] += 1
                            validation_results['errors'].append({
                                'file': label_path.name,
                                'line': line_num,
                                'error': f"Expected 5 values, got {len(parts)}"
                            })
                            continue

                        # Validate class_id is integer
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            validation_results['invalid_labels'] += 1
                            validation_results['errors'].append({
                                'file': label_path.name,
                                'line': line_num,
                                'error': "Class ID must be integer"
                            })
                            continue

                        # Validate coordinates are floats in range [0, 1]
                        try:
                            coords = [float(x) for x in parts[1:]]
                            if not all(0.0 <= x <= 1.0 for x in coords):
                                validation_results['invalid_labels'] += 1
                                validation_results['errors'].append({
                                    'file': label_path.name,
                                    'line': line_num,
                                    'error': "Coordinates must be in range [0, 1]"
                                })
                                continue
                        except ValueError:
                            validation_results['invalid_labels'] += 1
                            validation_results['errors'].append({
                                'file': label_path.name,
                                'line': line_num,
                                'error': "Coordinates must be floats"
                            })
                            continue

                        validation_results['valid_labels'] += 1

            except Exception as e:
                validation_results['errors'].append({
                    'file': label_path.name,
                    'line': 0,
                    'error': f"Failed to read file: {e}"
                })

        return validation_results

# Example usage
if __name__ == '__main__':
    # Status callback
    def status_callback(message, level):
        print(f"[{level.upper()}] {message}")

    # Create module
    module = AnnotationReviewModule("~/yolo_dataset", status_callback)

    # Get statistics
    print("\nDataset Statistics:")
    stats = module.get_dataset_statistics()
    print(f"Total images: {stats['total_images']}")
    print(f"Images with labels: {stats['images_with_labels']}")
    print(f"Total annotations: {stats['total_labels']}")
    print(f"\nClass distribution:")
    for class_name, count in sorted(stats['class_distribution'].items()):
        print(f"  {class_name}: {count}")

    # Validate labels
    print("\nValidating labels...")
    validation = module.validate_labels()
    print(f"Valid labels: {validation['valid_labels']}")
    print(f"Invalid labels: {validation['invalid_labels']}")
    if validation['errors']:
        print(f"\nErrors found:")
        for error in validation['errors'][:10]:  # Show first 10 errors
            print(f"  {error['file']}:{error['line']} - {error['error']}")
