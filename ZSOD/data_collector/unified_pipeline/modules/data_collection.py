#!/usr/bin/env python3

import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Callable
import threading

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.pipeline_config import CollectionConfig

class DataCollectionModule:
    """Module for managing data collection from ROS2 camera topics"""

    def __init__(self, config: CollectionConfig, status_callback: Optional[Callable] = None):
        """
        Initialize data collection module

        Args:
            config: Collection configuration
            status_callback: Optional callback function for status updates
        """
        self.config = config
        self.status_callback = status_callback
        self.process = None
        self.is_running = False
        self.stats = {
            'total_collected': 0,
            'class_counts': {},
            'start_time': None,
            'end_time': None
        }

        # Get paths relative to data_collector directory
        self.data_collector_dir = Path(__file__).parent.parent.parent
        if self.config.use_yolo11:
            self.collector_script = self.data_collector_dir / "run_yolo11_collector.py"
        else:
            self.collector_script = self.data_collector_dir / "run_dataset_collector.py"

    def _update_status(self, message: str, level: str = 'info'):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(message, level)

    def start_collection(self, test_mode: bool = False):
        """
        Start data collection process

        Args:
            test_mode: If True, only test detection without saving data
        """
        if self.is_running:
            self._update_status("Collection is already running", 'warning')
            return False

        try:
            # Prepare environment variables
            env = os.environ.copy()
            env['COLLECTOR_CONF_THRESHOLD'] = str(self.config.conf_threshold)
            env['COLLECTOR_IOU_THRESHOLD'] = str(self.config.iou_threshold)
            env['COLLECTOR_INTERVAL'] = str(self.config.collection_interval)
            env['COLLECTOR_MIN_DETECTIONS'] = str(self.config.min_detections)
            env['COLLECTOR_MAX_DETECTIONS'] = str(self.config.max_detections)
            env['COLLECTOR_DATASET_PATH'] = self.config.dataset_path
            env['COLLECTOR_IMAGE_TOPIC'] = self.config.image_topic
            env['COLLECTOR_TEST_MODE'] = str(test_mode)

            if self.config.use_yolo11:
                env['COLLECTOR_MODEL_PATH'] = self.config.yolo11_model_path
            else:
                env['COLLECTOR_MODEL_PATH'] = self.config.model_path

            # Prepare command
            cmd = ['python3', str(self.collector_script)]

            if test_mode:
                cmd.append('--test_mode')

            self._update_status(f"Starting data collection ({'test mode' if test_mode else 'collection mode'})...", 'info')
            self._update_status(f"Model: {'YOLO11' if self.config.use_yolo11 else 'YOLOE'}", 'info')
            self._update_status(f"Classes: {', '.join(self.config.target_classes[:5])}...", 'info')

            # Start subprocess
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid  # Create new process group
            )

            self.is_running = True
            self.stats['start_time'] = time.time()

            # Start output monitoring thread
            self.output_thread = threading.Thread(target=self._monitor_output, daemon=True)
            self.output_thread.start()

            self._update_status("Data collection started successfully", 'success')
            return True

        except Exception as e:
            self._update_status(f"Failed to start collection: {e}", 'error')
            return False

    def _monitor_output(self):
        """Monitor subprocess output for statistics"""
        if not self.process:
            return

        try:
            for line in self.process.stdout:
                line = line.strip()
                if line:
                    # Parse statistics from output
                    if "Saved dataset sample" in line:
                        self.stats['total_collected'] += 1
                        self._update_status(f"Collected: {self.stats['total_collected']}", 'info')

                    # Forward output to callback
                    self._update_status(line, 'debug')

        except Exception as e:
            self._update_status(f"Output monitoring error: {e}", 'error')

    def stop_collection(self):
        """Stop data collection process"""
        if not self.is_running:
            self._update_status("Collection is not running", 'warning')
            return False

        try:
            if self.process:
                # Send SIGTERM to entire process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for process to terminate
                self.process.wait(timeout=5)

                self.stats['end_time'] = time.time()
                duration = self.stats['end_time'] - self.stats['start_time']

                self._update_status(f"Collection stopped. Duration: {duration:.1f}s", 'success')
                self._update_status(f"Total collected: {self.stats['total_collected']}", 'info')

            self.is_running = False
            self.process = None
            return True

        except subprocess.TimeoutExpired:
            # Force kill if not responding
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.is_running = False
            self.process = None
            self._update_status("Collection force stopped", 'warning')
            return True

        except Exception as e:
            self._update_status(f"Failed to stop collection: {e}", 'error')
            return False

    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        return self.stats.copy()

    def get_dataset_info(self) -> Dict:
        """Get information about collected dataset"""
        dataset_path = Path(self.config.dataset_path).expanduser()
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        info = {
            'path': str(dataset_path),
            'exists': dataset_path.exists(),
            'images_count': 0,
            'labels_count': 0
        }

        if images_dir.exists():
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            images = []
            for ext in extensions:
                images.extend(list(images_dir.glob(ext)))
            info['images_count'] = len(images)

        if labels_dir.exists():
            labels = list(labels_dir.glob('*.txt'))
            info['labels_count'] = len(labels)

        return info

    def validate_dataset(self) -> bool:
        """Validate that dataset directory and structure are correct"""
        dataset_path = Path(self.config.dataset_path).expanduser()

        if not dataset_path.exists():
            self._update_status(f"Dataset directory does not exist: {dataset_path}", 'error')
            return False

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        if not images_dir.exists():
            self._update_status(f"Images directory missing: {images_dir}", 'error')
            return False

        if not labels_dir.exists():
            self._update_status(f"Labels directory missing: {labels_dir}", 'error')
            return False

        return True

# Example usage
if __name__ == '__main__':
    from config.pipeline_config import CollectionConfig

    # Create test configuration
    config = CollectionConfig(
        target_classes=["chair", "table", "bed"],
        conf_threshold=0.6,
        collection_interval=2.0,
        dataset_path="~/yolo_dataset"
    )

    # Status callback
    def status_callback(message, level):
        print(f"[{level.upper()}] {message}")

    # Create module
    module = DataCollectionModule(config, status_callback)

    # Test mode
    print("Starting test mode...")
    module.start_collection(test_mode=True)

    # Run for 10 seconds
    time.sleep(10)

    # Stop collection
    module.stop_collection()

    # Show statistics
    print("\nStatistics:", module.get_statistics())
    print("Dataset info:", module.get_dataset_info())
