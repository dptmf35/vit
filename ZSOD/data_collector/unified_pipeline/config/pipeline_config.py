#!/usr/bin/env python3

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import yaml

@dataclass
class CollectionConfig:
    """Configuration for data collection phase"""
    target_classes: List[str]
    conf_threshold: float = 0.6
    iou_threshold: float = 0.4
    collection_interval: float = 2.0
    min_detections: int = 1
    max_detections: int = 50
    image_topic: str = '/stereo_image_color'
    model_path: str = 'yoloe-11m-seg.pt'
    dataset_path: str = '~/yolo_dataset'
    use_yolo11: bool = False  # False for YOLOE, True for YOLO11
    yolo11_model_path: str = 'train_model/training_output/train/weights/best.pt'

@dataclass
class TrainingConfig:
    """Configuration for training phase"""
    model_size: str = 'yolo11s.pt'  # yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    epochs: int = 100
    batch_size: Optional[int] = None  # Auto-determined if None
    image_size: int = 640
    validation_ratio: float = 0.2
    output_dir: str = 'training_output'

@dataclass
class EvaluationConfig:
    """Configuration for evaluation phase"""
    conf_threshold: float = 0.25
    test_dataset_path: Optional[str] = None  # None to use validation set
    save_results: bool = True
    output_dir: str = 'evaluation_results'

@dataclass
class DeploymentConfig:
    """Configuration for deployment phase"""
    model_path: str = 'train_model/training_output/train/weights/best.pt'
    camera_topic: str = '/stereo_image_color'
    output_topic: str = '/custom_yolo/annotated_image'
    detection_topic: str = '/custom_yolo/detections'
    bbox_topic: str = '/custom_yolo/bounding_boxes'
    conf_threshold: float = 0.5
    publish_annotated: bool = True
    save_detections: bool = False

@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    collection: CollectionConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    deployment: DeploymentConfig

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            'collection': asdict(self.collection),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation),
            'deployment': asdict(self.deployment)
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls(
            collection=CollectionConfig(**config_dict['collection']),
            training=TrainingConfig(**config_dict['training']),
            evaluation=EvaluationConfig(**config_dict['evaluation']),
            deployment=DeploymentConfig(**config_dict['deployment'])
        )

    @classmethod
    def create_default(cls, target_classes: List[str]):
        """Create default configuration with specified target classes"""
        return cls(
            collection=CollectionConfig(target_classes=target_classes),
            training=TrainingConfig(),
            evaluation=EvaluationConfig(),
            deployment=DeploymentConfig()
        )

class ConfigManager:
    """Manager for pipeline configuration"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / 'pipeline_config.json'

    def save_config(self, config: PipelineConfig):
        """Save configuration to file"""
        config.save_to_file(self.config_file)

    def load_config(self) -> Optional[PipelineConfig]:
        """Load configuration from file"""
        if self.config_file.exists():
            return PipelineConfig.load_from_file(self.config_file)
        return None

    def create_default_config(self, target_classes: List[str]) -> PipelineConfig:
        """Create and save default configuration"""
        config = PipelineConfig.create_default(target_classes)
        self.save_config(config)
        return config

    def get_default_classes(self) -> List[str]:
        """Get default class list"""
        return [
            "air purifier", "bed", "cabinet", "carpet", "chair", "closet",
            "countertop", "desk", "dinningtable", "door", "fridge", "lamp",
            "mirror", "piano", "plant", "shelf", "sidetable", "sofa",
            "table", "tv", "tv stand", "vanity"
        ]

# Example usage
if __name__ == '__main__':
    # Create default configuration
    manager = ConfigManager()
    default_classes = manager.get_default_classes()
    config = manager.create_default_config(default_classes)

    print("Default configuration created:")
    print(f"Collection: {config.collection}")
    print(f"Training: {config.training}")
    print(f"Evaluation: {config.evaluation}")
    print(f"Deployment: {config.deployment}")
