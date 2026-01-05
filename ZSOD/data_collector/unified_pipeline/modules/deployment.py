#!/usr/bin/env python3

import sys
import os
import signal
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict
import tempfile

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.pipeline_config import DeploymentConfig

class DeploymentModule:
    """Module for deploying trained models as ROS2 nodes"""

    def __init__(self, config: DeploymentConfig, status_callback: Optional[Callable] = None):
        """
        Initialize deployment module

        Args:
            config: Deployment configuration
            status_callback: Optional callback function for status updates
        """
        self.config = config
        self.status_callback = status_callback
        self.process = None
        self.is_running = False

        # Create temporary deployment script
        self.deployment_script = self._create_deployment_script()

    def _update_status(self, message: str, level: str = 'info'):
        """Update status via callback"""
        if self.status_callback:
            self.status_callback(message, level)

    def _create_deployment_script(self) -> Path:
        """Create ROS2 deployment node script"""
        script_content = '''#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path
import os

class CustomYOLODetector(Node):
    def __init__(self):
        super().__init__('custom_yolo_detector')

        # Get parameters from environment
        model_path = os.getenv('DEPLOY_MODEL_PATH')
        camera_topic = os.getenv('DEPLOY_CAMERA_TOPIC', '/stereo_image_color')
        output_topic = os.getenv('DEPLOY_OUTPUT_TOPIC', '/custom_yolo/annotated_image')
        detection_topic = os.getenv('DEPLOY_DETECTION_TOPIC', '/custom_yolo/detections')
        bbox_topic = os.getenv('DEPLOY_BBOX_TOPIC', '/custom_yolo/bounding_boxes')
        conf_threshold = float(os.getenv('DEPLOY_CONF_THRESHOLD', '0.5'))
        publish_annotated = os.getenv('DEPLOY_PUBLISH_ANNOTATED', 'True').lower() == 'true'
        save_detections = os.getenv('DEPLOY_SAVE_DETECTIONS', 'False').lower() == 'true'

        # Load model
        self.load_model(model_path)

        # CV Bridge
        self.bridge = CvBridge()
        self.confidence_threshold = conf_threshold
        self.publish_annotated = publish_annotated
        self.save_detections = save_detections

        # Subscribers
        self.image_subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )

        # Publishers
        self.detection_publisher = self.create_publisher(
            String,
            detection_topic,
            10
        )

        self.bbox_publisher = self.create_publisher(
            Float32MultiArray,
            bbox_topic,
            10
        )

        if self.publish_annotated:
            self.annotated_image_publisher = self.create_publisher(
                Image,
                output_topic,
                10
            )

        # Detection counter
        self.detection_count = 0

        # Output directory for saving
        if self.save_detections:
            self.output_dir = Path("deployment_detections")
            self.output_dir.mkdir(exist_ok=True)

        self.get_logger().info("Custom YOLO Detector initialized")
        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Camera topic: {camera_topic}")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"Classes: {self.class_names}")

    def load_model(self, model_path):
        """Load the trained YOLO model"""
        try:
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.getcwd(), model_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = YOLO(model_path)
            self.class_names = self.model.names

            self.get_logger().info(f"Successfully loaded model: {model_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

    def image_callback(self, msg):
        """Process incoming images"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run detection
            results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)

            if not results:
                return

            result = results[0]

            # Prepare annotated image
            annotated_image = cv_image.copy()
            detections = []
            bboxes = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Extract detection info
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    class_name = self.class_names[cls] if isinstance(self.class_names, dict) else self.class_names[cls]

                    detection = {
                        'class_id': cls,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': xyxy.tolist()
                    }
                    detections.append(detection)
                    bboxes.extend([cls, conf] + xyxy.tolist())

                    # Draw on annotated image
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = self.get_class_color(cls)

                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                    cv2.rectangle(annotated_image,
                                (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0] + 10, y1),
                                color, -1)

                    cv2.putText(annotated_image, label,
                              (x1 + 5, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Publish detections
            if detections:
                detection_msg = String()
                detection_msg.data = json.dumps(detections)
                self.detection_publisher.publish(detection_msg)

                # Publish bounding boxes
                bbox_msg = Float32MultiArray()
                bbox_msg.data = bboxes
                self.bbox_publisher.publish(bbox_msg)

                self.get_logger().info(f"Detected {len(detections)} objects", throttle_duration_sec=1.0)

            # Publish annotated image
            if self.publish_annotated:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                self.annotated_image_publisher.publish(annotated_msg)

            # Save detection if enabled
            if self.save_detections and detections:
                self.detection_count += 1
                output_path = self.output_dir / f"detection_{self.detection_count:06d}.jpg"
                cv2.imwrite(str(output_path), annotated_image)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def get_class_color(self, class_id):
        """Generate color for class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 165, 0),
            (255, 20, 147), (0, 191, 255), (255, 69, 0), (50, 205, 50), (138, 43, 226)
        ]
        return colors[class_id % len(colors)]

def main(args=None):
    rclpy.init(args=args)
    node = CustomYOLODetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
'''

        # Write script to temporary file
        temp_script = Path(tempfile.mktemp(suffix='.py'))
        temp_script.write_text(script_content)
        temp_script.chmod(0o755)

        return temp_script

    def start_deployment(self):
        """Start ROS2 deployment node"""
        if self.is_running:
            self._update_status("Deployment is already running", 'warning')
            return False

        # Validate model exists
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            self._update_status(f"Model not found: {model_path}", 'error')
            return False

        try:
            # Prepare environment variables
            env = os.environ.copy()
            env['DEPLOY_MODEL_PATH'] = str(model_path.absolute())
            env['DEPLOY_CAMERA_TOPIC'] = self.config.camera_topic
            env['DEPLOY_OUTPUT_TOPIC'] = self.config.output_topic
            env['DEPLOY_DETECTION_TOPIC'] = self.config.detection_topic
            env['DEPLOY_BBOX_TOPIC'] = self.config.bbox_topic
            env['DEPLOY_CONF_THRESHOLD'] = str(self.config.conf_threshold)
            env['DEPLOY_PUBLISH_ANNOTATED'] = str(self.config.publish_annotated)
            env['DEPLOY_SAVE_DETECTIONS'] = str(self.config.save_detections)

            self._update_status("Starting ROS2 deployment node...", 'info')
            self._update_status(f"Model: {model_path}", 'info')
            self._update_status(f"Camera: {self.config.camera_topic}", 'info')
            self._update_status(f"Output: {self.config.output_topic}", 'info')

            # Start ROS2 node
            cmd = ['python3', str(self.deployment_script)]

            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )

            self.is_running = True
            self._update_status("Deployment node started successfully", 'success')
            return True

        except Exception as e:
            self._update_status(f"Failed to start deployment: {e}", 'error')
            return False

    def stop_deployment(self):
        """Stop ROS2 deployment node"""
        if not self.is_running:
            self._update_status("Deployment is not running", 'warning')
            return False

        try:
            if self.process:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for process to terminate
                self.process.wait(timeout=5)

                self._update_status("Deployment stopped", 'success')

            self.is_running = False
            self.process = None
            return True

        except subprocess.TimeoutExpired:
            # Force kill
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.is_running = False
            self.process = None
            self._update_status("Deployment force stopped", 'warning')
            return True

        except Exception as e:
            self._update_status(f"Failed to stop deployment: {e}", 'error')
            return False

    def get_status(self) -> Dict:
        """Get deployment status"""
        return {
            'is_running': self.is_running,
            'model_path': self.config.model_path,
            'camera_topic': self.config.camera_topic,
            'output_topic': self.config.output_topic,
            'detection_topic': self.config.detection_topic
        }

# Example usage
if __name__ == '__main__':
    from config.pipeline_config import DeploymentConfig

    # Status callback
    def status_callback(message, level):
        print(f"[{level.upper()}] {message}")

    # Create configuration
    config = DeploymentConfig(
        model_path='train_model/training_output/train/weights/best.pt',
        camera_topic='/stereo_image_color',
        conf_threshold=0.5
    )

    # Create module
    module = DeploymentModule(config, status_callback)

    print("Deployment configuration:")
    print(f"Status: {module.get_status()}")
