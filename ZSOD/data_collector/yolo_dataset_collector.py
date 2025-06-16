#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import time
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLOE

class YOLODatasetCollector(Node):
    def __init__(self):
        super().__init__('yolo_dataset_collector')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize YOLOE model
        self.model = YOLOE("yoloe-11s-seg.pt")
        
        # Set text prompt to detect objects
        self.class_names = ["table", "fridge", "chair", "dish", "gas stove", "closet", "lamp", "curtain", "nightstand",
                           "microwave", "tv", "sofa", "shelf", "window", "door", "bed", "cabinet", "plant", "computer"]
        self.model.set_classes(self.class_names, self.model.get_text_pe(self.class_names))
        
        # Detection parameters - 더 높은 threshold 설정으로 고품질 데이터 수집
        self.conf_threshold = float(os.getenv('COLLECTOR_CONF_THRESHOLD', 0.6))
        self.iou_threshold = float(os.getenv('COLLECTOR_IOU_THRESHOLD', 0.4))
        
        # Collection control parameters
        self.collection_interval = float(os.getenv('COLLECTOR_INTERVAL', 2.0))
        self.last_collection_time = 0.0
        self.min_detections = int(os.getenv('COLLECTOR_MIN_DETECTIONS', 1))
        self.max_detections_per_image = int(os.getenv('COLLECTOR_MAX_DETECTIONS', 50))
        
        # Dataset directory setup
        self.setup_dataset_directories()
        
        # Collection statistics
        self.total_collected = 0
        self.collected_by_class = {name: 0 for name in self.class_names}
        
        # Subscribe to image topic (configurable)
        image_topic = os.getenv('COLLECTOR_IMAGE_TOPIC', '/stereo_image_color')
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        # Create publisher for detection results (optional for visualization)
        self.result_pub = self.create_publisher(
            Image,
            '/yolo_collector_result',
            10
        )
        
        self.get_logger().info("YOLO Dataset Collector initialized")
        self.get_logger().info(f"Collection parameters: conf={self.conf_threshold}, iou={self.iou_threshold}")
        self.get_logger().info(f"Collection interval: {self.collection_interval}s")
        self.get_logger().info(f"Target classes: {', '.join(self.class_names)}")
    
    def setup_dataset_directories(self):
        """Setup dataset directory structure"""
        dataset_path = os.getenv('COLLECTOR_DATASET_PATH', '~/yolo_dataset')
        self.dataset_root = os.path.expanduser(dataset_path)
        self.images_dir = os.path.join(self.dataset_root, "images")
        self.labels_dir = os.path.join(self.dataset_root, "labels")
        
        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Create dataset.yaml file for YOLO training
        yaml_content = f"""train: {self.images_dir}
val: {self.images_dir}
test: {self.images_dir}

nc: {len(self.class_names)}
names: {self.class_names}
"""
        yaml_path = os.path.join(self.dataset_root, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        self.get_logger().info(f"Dataset directory created at: {self.dataset_root}")
    
    def normalize_bbox(self, bbox, img_width, img_height):
        """Convert bbox to YOLO format (normalized coordinates)"""
        x1, y1, x2, y2 = bbox
        
        # Calculate center point and dimensions
        center_x = (x1 + x2) / 2.0 / img_width
        center_y = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return center_x, center_y, width, height
    
    def should_collect_data(self):
        """Check if enough time has passed since last collection"""
        current_time = time.time()
        if current_time - self.last_collection_time >= self.collection_interval:
            self.last_collection_time = current_time
            return True
        return False
    
    def save_dataset_sample(self, cv_image, detections):
        """Save image and corresponding YOLO annotation file"""
        if len(detections) < self.min_detections:
            return False
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_filename = f"img_{timestamp}.jpg"
        label_filename = f"img_{timestamp}.txt"
        
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, label_filename)
        
        # Save image
        cv2.imwrite(image_path, cv_image)
        
        # Save YOLO format annotation
        img_height, img_width = cv_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for detection in detections:
                class_id, bbox, confidence = detection
                
                # Normalize bbox coordinates
                center_x, center_y, width, height = self.normalize_bbox(bbox, img_width, img_height)
                
                # Write YOLO format: class_id center_x center_y width height
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                # Update statistics
                class_name = self.class_names[class_id]
                self.collected_by_class[class_name] += 1
        
        self.total_collected += 1
        self.get_logger().info(f"Saved dataset sample {self.total_collected}: {image_filename}")
        
        # Log collection statistics every 10 samples
        if self.total_collected % 10 == 0:
            self.log_collection_stats()
        
        return True
    
    def log_collection_stats(self):
        """Log collection statistics"""
        self.get_logger().info(f"=== Collection Statistics (Total: {self.total_collected}) ===")
        for class_name, count in self.collected_by_class.items():
            if count > 0:
                self.get_logger().info(f"  {class_name}: {count}")
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLOE prediction with higher thresholds for quality data
            results = self.model.predict(
                cv_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections_per_image,
                agnostic_nms=False,
                verbose=False  # Reduce prediction logging
            )
            
            # Process results for dataset collection
            if len(results) > 0 and results[0].boxes is not None:
                detections = []
                
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    # Extract bbox coordinates (x1, y1, x2, y2)
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    detections.append((class_id, bbox, confidence))
                
                # Collect data if conditions are met
                if detections and self.should_collect_data():
                    success = self.save_dataset_sample(cv_image, detections)
                    
                    if success:
                        # Log detected objects for this collection
                        detected_classes = [self.class_names[det[0]] for det in detections]
                        unique_classes = list(set(detected_classes))
                        self.get_logger().info(f"Collected data with classes: {', '.join(unique_classes)}")
                
                # Publish visualization (optional)
                if self.result_pub.get_subscription_count() > 0:
                    annotated_image = results[0].plot()
                    result_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                    result_msg.header = msg.header
                    self.result_pub.publish(result_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    node = YOLODatasetCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO Dataset Collector...")
        node.log_collection_stats()  # Final statistics
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 