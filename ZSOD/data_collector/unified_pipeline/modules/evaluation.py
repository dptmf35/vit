#!/usr/bin/env python3

import sys
import cv2
from pathlib import Path
from typing import Optional, Callable, Dict, List
from ultralytics import YOLO

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.pipeline_config import EvaluationConfig

class EvaluationModule:
    """Module for evaluating trained YOLO models"""

    def __init__(self, config: EvaluationConfig, model_path: str, status_callback: Optional[Callable] = None):
        """
        Initialize evaluation module

        Args:
            config: Evaluation configuration
            model_path: Path to trained model
            status_callback: Optional callback function for status updates
        """
        self.config = config
        self.model_path = Path(model_path)
        self.status_callback = status_callback
        self.model = None
        self.class_names = []

        # Output directory
        self.output_dir = Path(self.config.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _update_status(self, message: str, level: str = 'info'):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(message, level)

    def load_model(self) -> bool:
        """Load the trained model"""
        if not self.model_path.exists():
            self._update_status(f"Model not found: {self.model_path}", 'error')
            return False

        try:
            self._update_status(f"Loading model: {self.model_path}", 'info')
            self.model = YOLO(str(self.model_path))

            # Get class names
            self.class_names = self.model.names
            if isinstance(self.class_names, dict):
                self.class_names = list(self.class_names.values())

            self._update_status(f"Model loaded with {len(self.class_names)} classes", 'success')
            return True

        except Exception as e:
            self._update_status(f"Failed to load model: {e}", 'error')
            return False

    def evaluate_on_dataset(self, dataset_yaml_path: str) -> Dict:
        """
        Evaluate model on a dataset

        Args:
            dataset_yaml_path: Path to dataset.yaml file

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model:
            if not self.load_model():
                return {}

        try:
            self._update_status("Running evaluation...", 'info')

            # Run validation
            results = self.model.val(
                data=dataset_yaml_path,
                conf=self.config.conf_threshold,
                save_json=True,
                plots=True,
                project=str(self.output_dir),
                name='eval'
            )

            # Extract metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'class_metrics': {}
            }

            # Per-class metrics
            if hasattr(results.box, 'maps'):
                for i, ap in enumerate(results.box.maps):
                    if i < len(self.class_names):
                        metrics['class_metrics'][self.class_names[i]] = {
                            'AP': float(ap)
                        }

            self._update_status(f"Evaluation complete: mAP50={metrics['mAP50']:.3f}", 'success')
            return metrics

        except Exception as e:
            self._update_status(f"Evaluation failed: {e}", 'error')
            return {}

    def test_on_image(self, image_path: str, save_result: bool = True) -> Optional[Dict]:
        """
        Test model on a single image

        Args:
            image_path: Path to image file
            save_result: Whether to save annotated result

        Returns:
            Detection results dictionary
        """
        if not self.model:
            if not self.load_model():
                return None

        image_path = Path(image_path)
        if not image_path.exists():
            self._update_status(f"Image not found: {image_path}", 'error')
            return None

        try:
            # Run prediction
            results = self.model.predict(
                str(image_path),
                conf=self.config.conf_threshold,
                save=False,
                verbose=False
            )

            if not results:
                return None

            result = results[0]

            # Load image
            image = cv2.imread(str(image_path))
            annotated_image = image.copy()

            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract info
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"

                    detections.append({
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': xyxy.tolist()
                    })

                    # Draw on image
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = self._get_class_color(cls)

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_name}: {conf:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    cv2.rectangle(annotated_image,
                                (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0] + 10, y1),
                                color, -1)

                    cv2.putText(annotated_image, label,
                              (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save result
            output_path = None
            if save_result and self.config.save_results:
                output_path = self.output_dir / f"result_{image_path.name}"
                cv2.imwrite(str(output_path), annotated_image)

            return {
                'image_path': str(image_path),
                'detections': detections,
                'annotated_image': annotated_image,
                'output_path': str(output_path) if output_path else None
            }

        except Exception as e:
            self._update_status(f"Test failed: {e}", 'error')
            return None

    def test_on_directory(self, directory_path: str) -> Dict:
        """
        Test model on all images in a directory

        Args:
            directory_path: Path to directory containing images

        Returns:
            Summary statistics
        """
        directory = Path(directory_path)
        if not directory.exists():
            self._update_status(f"Directory not found: {directory}", 'error')
            return {}

        # Find images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(list(directory.glob(ext)))

        if not images:
            self._update_status("No images found", 'error')
            return {}

        self._update_status(f"Testing on {len(images)} images...", 'info')

        results = []
        for img_path in images:
            result = self.test_on_image(str(img_path), save_result=self.config.save_results)
            if result:
                results.append(result)

        # Summary statistics
        total_detections = sum(len(r['detections']) for r in results)
        images_with_detections = sum(1 for r in results if r['detections'])

        summary = {
            'total_images': len(results),
            'images_with_detections': images_with_detections,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(results) if results else 0
        }

        self._update_status(
            f"Testing complete: {summary['images_with_detections']}/{summary['total_images']} images with detections",
            'success'
        )

        return summary

    def _get_class_color(self, class_id: int):
        """Get color for class visualization"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 165, 0),
            (255, 20, 147), (0, 191, 255), (255, 69, 0), (50, 205, 50), (138, 43, 226)
        ]
        return colors[class_id % len(colors)]

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if not self.model:
            return {}

        return {
            'model_path': str(self.model_path),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_loaded': True
        }

# Example usage
if __name__ == '__main__':
    from config.pipeline_config import EvaluationConfig

    # Status callback
    def status_callback(message, level):
        print(f"[{level.upper()}] {message}")

    # Create configuration
    config = EvaluationConfig(
        conf_threshold=0.25,
        save_results=True
    )

    # Create module
    model_path = "train_model/training_output/train/weights/best.pt"
    module = EvaluationModule(config, model_path, status_callback)

    # Load model
    if module.load_model():
        print("\nModel Info:")
        info = module.get_model_info()
        print(f"Classes: {info['class_names']}")
