#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import time
from datetime import datetime
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLOE
import threading

class YOLODatasetCollector(Node):
    def __init__(self):
        super().__init__('yolo_dataset_collector')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize YOLOE model
        self.model = YOLOE("yoloe-11m-seg.pt")
        
        # Set text prompt to detect objects (sorted alphabetically)
        self.class_names = ["air purifier", "bed", "cabinet", "carpet", "chair", "closet", "countertop", "curtain", "desk", "door", "fridge",
                           "gas stove", "hanger", "kitchen cart", "lamp", "nightstand", "plant", "shelf", "sofa", "table", 
                           "tv", "window", "vanity"]
        self.model.set_classes(self.class_names, self.model.get_text_pe(self.class_names))
        
        # Detection parameters - 더 높은 threshold 설정으로 고품질 데이터 수집
        self.conf_threshold = float(os.getenv('COLLECTOR_CONF_THRESHOLD', 0.6))
        self.iou_threshold = float(os.getenv('COLLECTOR_IOU_THRESHOLD', 0.4))
        
        # Collection control parameters
        self.collection_interval = float(os.getenv('COLLECTOR_INTERVAL', 2.0))
        self.last_collection_time = 0.0
        self.min_detections = int(os.getenv('COLLECTOR_MIN_DETECTIONS', 1))
        self.max_detections_per_image = int(os.getenv('COLLECTOR_MAX_DETECTIONS', 50))
        
        # Test mode - only publish detection results, no data collection
        self.test_mode = os.getenv('COLLECTOR_TEST_MODE', 'False').lower() == 'true'
        
        # Dataset directory setup (skip in test mode)
        if not self.test_mode:
            self.setup_dataset_directories()
        else:
            self.get_logger().info("Test mode enabled - skipping dataset directory setup")
        
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
        
        # Create publisher for rviz detection visualization
        self.rviz_pub = self.create_publisher(
            Image,
            '/yolo_detection_rviz',
            10
        )
        
        # Create mode status publisher
        self.mode_pub = self.create_publisher(
            String,
            '/collector_mode_status',
            10
        )
        
        # Create service to toggle between test and collection mode
        self.mode_service = self.create_service(
            SetBool,
            '/toggle_collection_mode',
            self.toggle_mode_callback
        )
        
        # Start keyboard input thread for interactive mode switching
        self.keyboard_thread = threading.Thread(target=self.keyboard_input_handler, daemon=True)
        self.keyboard_thread.start()
        
        if self.test_mode:
            self.get_logger().info("🔍 YOLO Dataset Collector initialized - TEST MODE")
            self.get_logger().info("Detection results will be published to /yolo_detection_rviz")
            self.get_logger().info("NO DATA COLLECTION will occur")
        else:
            self.get_logger().info("YOLO Dataset Collector initialized - DATA COLLECTION MODE")
            
        self.get_logger().info(f"Detection parameters: conf={self.conf_threshold}, iou={self.iou_threshold}")
        self.get_logger().info(f"Collection interval: {self.collection_interval}s")
        self.get_logger().info(f"Target classes: {', '.join(self.class_names)}")
        self.get_logger().info("")
        self.get_logger().info("=== Mode Switching Controls ===")
        self.get_logger().info("Keyboard: Press 't' for Test Mode, 'c' for Collection Mode, 's' for Status")
        self.get_logger().info("ROS2 Service: ros2 service call /toggle_collection_mode std_srvs/srv/SetBool \"{data: true}\"")
        self.get_logger().info("Status Topic: ros2 topic echo /collector_mode_status")
        self.get_logger().info("================================")
        
        # Publish initial mode status
        self.publish_mode_status()
    
    def setup_dataset_directories(self):
        """Setup dataset directory structure"""
        dataset_path = os.getenv('COLLECTOR_DATASET_PATH', '~/yolo_dataset')
        self.dataset_root = os.path.expanduser(dataset_path)
        self.images_dir = os.path.join(self.dataset_root, "images")
        self.labels_dir = os.path.join(self.dataset_root, "labels")
        self.visualizations_dir = os.path.join(self.dataset_root, "visualizations")
        
        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
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
        self.get_logger().info(f"Visualizations will be saved to: {self.visualizations_dir}")
    
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
    
    def save_dataset_sample(self, cv_image, detections, annotated_image):
        """Save image, annotation file, and visualization"""
        if len(detections) < self.min_detections:
            return False
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_filename = f"img_{timestamp}.jpg"
        label_filename = f"img_{timestamp}.txt"
        vis_filename = f"vis_{timestamp}.jpg"
        
        image_path = os.path.join(self.images_dir, image_filename)
        label_path = os.path.join(self.labels_dir, label_filename)
        vis_path = os.path.join(self.visualizations_dir, vis_filename)
        
        # Save original image
        cv2.imwrite(image_path, cv_image)
        
        # Save annotated visualization image
        cv2.imwrite(vis_path, annotated_image)
        
        # Save YOLO format annotation
        img_height, img_width = cv_image.shape[:2]
        
        with open(label_path, 'w') as f:
            for detection in detections:
                class_id, bbox, confidence = detection
                
                # Validate bbox coordinates
                x1, y1, x2, y2 = bbox
                if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
                    self.get_logger().warn(f"Invalid bbox detected: {bbox} for image size {img_width}x{img_height}")
                    continue
                
                # Normalize bbox coordinates
                center_x, center_y, width, height = self.normalize_bbox(bbox, img_width, img_height)
                
                # Additional validation for normalized coordinates
                if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    self.get_logger().warn(f"Invalid normalized bbox: {center_x}, {center_y}, {width}, {height}")
                    continue
                
                # Write YOLO format: class_id center_x center_y width height
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                # Update statistics
                class_name = self.class_names[class_id]
                self.collected_by_class[class_name] += 1
        
        self.total_collected += 1
        self.get_logger().info(f"Saved dataset sample {self.total_collected}: {image_filename}")
        self.get_logger().info(f"Saved visualization: {vis_filename}")
        
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
    
    def toggle_mode_callback(self, request, response):
        """ROS2 service callback to toggle collection mode"""
        if request.data:  # True = enable collection mode
            if self.test_mode:
                self.switch_to_collection_mode()
                response.success = True
                response.message = "Switched to COLLECTION MODE"
            else:
                response.success = False
                response.message = "Already in COLLECTION MODE"
        else:  # False = enable test mode
            if not self.test_mode:
                self.switch_to_test_mode()
                response.success = True
                response.message = "Switched to TEST MODE"
            else:
                response.success = False
                response.message = "Already in TEST MODE"
        
        return response
    
    def switch_to_test_mode(self):
        """Switch to test mode (detection only)"""
        self.test_mode = True
        self.get_logger().info("🔍 SWITCHED TO TEST MODE - Detection only, no data collection")
        self.publish_mode_status()
    
    def switch_to_collection_mode(self):
        """Switch to collection mode"""
        self.test_mode = False
        
        # Setup dataset directories if not already done
        if not hasattr(self, 'dataset_root'):
            self.setup_dataset_directories()
        
        self.get_logger().info("💾 SWITCHED TO COLLECTION MODE - Data will be saved")
        self.publish_mode_status()
    
    def publish_mode_status(self):
        """Publish current mode status"""
        mode_msg = String()
        if self.test_mode:
            mode_msg.data = "TEST_MODE"
        else:
            mode_msg.data = "COLLECTION_MODE"
        self.mode_pub.publish(mode_msg)
    
    def keyboard_input_handler(self):
        """Handle keyboard input for mode switching"""
        self.get_logger().info("Keyboard input thread started. Press 't', 'c', or 's' to control mode.")
        
        try:
            while rclpy.ok():
                try:
                    # Non-blocking input with timeout
                    import select
                    import sys
                    
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower().strip()
                        
                        if key == 't':
                            if not self.test_mode:
                                self.switch_to_test_mode()
                            else:
                                self.get_logger().info("Already in TEST MODE")
                        elif key == 'c':
                            if self.test_mode:
                                self.switch_to_collection_mode()
                            else:
                                self.get_logger().info("Already in COLLECTION MODE")
                        elif key == 's':
                            current_mode = "TEST MODE" if self.test_mode else "COLLECTION MODE"
                            self.get_logger().info(f"Current mode: {current_mode}")
                            if not self.test_mode:
                                self.log_collection_stats()
                        elif key == 'q':
                            self.get_logger().info("Shutting down...")
                            rclpy.shutdown()
                            break
                            
                except Exception as e:
                    # Ignore input errors
                    pass
                    
                time.sleep(0.1)
                
        except Exception as e:
            self.get_logger().error(f"Keyboard input handler error: {str(e)}")
    
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
                
                # Generate annotated image
                annotated_image = results[0].plot()
                
                # Collect data if conditions are met (skip in test mode)
                if not self.test_mode and detections and self.should_collect_data():
                    success = self.save_dataset_sample(cv_image, detections, annotated_image)
                    
                    if success:
                        # Log detected objects for this collection
                        detected_classes = [self.class_names[det[0]] for det in detections]
                        unique_classes = list(set(detected_classes))
                        self.get_logger().info(f"Collected data with classes: {', '.join(unique_classes)}")
                elif self.test_mode and detections:
                    # In test mode, just log detected objects without saving
                    detected_classes = [self.class_names[det[0]] for det in detections]
                    unique_classes = list(set(detected_classes))
                    self.get_logger().info(f"🔍 TEST MODE - Detected classes: {', '.join(unique_classes)} (not saved)")
                
                # Always publish to rviz for visualization
                try:
                    rviz_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                    rviz_msg.header = msg.header
                    rviz_msg.header.frame_id = "camera_link"  # Set appropriate frame_id for rviz
                    self.rviz_pub.publish(rviz_msg)
                except Exception as e:
                    self.get_logger().error(f"Failed to publish rviz visualization: {str(e)}")
                    
            else:
                # No detections - still publish empty visualization to rviz
                try:
                    rviz_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                    rviz_msg.header = msg.header
                    rviz_msg.header.frame_id = "camera_link"
                    self.rviz_pub.publish(rviz_msg)
                except Exception as e:
                    self.get_logger().error(f"Failed to publish rviz image: {str(e)}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    node = YOLODatasetCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.test_mode:
            node.get_logger().info("🔍 Shutting down YOLO Dataset Collector (TEST MODE)...")
        else:
            node.get_logger().info("Shutting down YOLO Dataset Collector...")
            node.log_collection_stats()  # Final statistics
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 