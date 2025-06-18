#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
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
        
        # Parameters
        self.declare_parameter('model_path', 'train_model/training_output/train/weights/best.pt')
        self.declare_parameter('camera_topic', '/stereo_image_color')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('save_detections', False)
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.publish_annotated = self.get_parameter('publish_annotated').get_parameter_value().bool_value
        self.save_detections = self.get_parameter('save_detections').get_parameter_value().bool_value
        
        # Initialize YOLO model
        self.load_model(model_path)
        
        # CV Bridge
        self.bridge = CvBridge()
        
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
            '/custom_yolo/detections',
            10
        )
        
        self.bbox_publisher = self.create_publisher(
            Float32MultiArray,
            '/custom_yolo/bounding_boxes',
            10
        )
        
        if self.publish_annotated:
            self.annotated_image_publisher = self.create_publisher(
                Image,
                '/custom_yolo/annotated_image',
                10
            )
        
        # Detection counter for saving
        self.detection_count = 0
        
        # Create output directory if saving detections
        if self.save_detections:
            self.output_dir = Path("custom_yolo_detections")
            self.output_dir.mkdir(exist_ok=True)
        
        self.get_logger().info(f"Custom YOLO Detector initialized")
        self.get_logger().info(f"Model: {model_path}")
        self.get_logger().info(f"Camera topic: {camera_topic}")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"Classes: {self.class_names}")
    
    def load_model(self, model_path):
        """Load the custom trained YOLO model"""
        try:
            # Convert relative path to absolute path
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.getcwd(), model_path)
            
            if not os.path.exists(model_path):
                self.get_logger().error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            
            self.get_logger().info(f"Successfully loaded model: {model_path}")
            self.get_logger().info(f"Model classes: {list(self.class_names.values())}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise e
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO detection
            results = self.model(cv_image, conf=self.confidence_threshold)
            
            # Process detections
            detections = self.process_detections(results[0], cv_image)
            
            # Publish detection results
            self.publish_detections(detections, msg.header.stamp)
            
            # Publish annotated image if enabled
            if self.publish_annotated:
                annotated_image = self.draw_detections(cv_image.copy(), detections)
                self.publish_annotated_image(annotated_image, msg.header)
            
            # Save detection if enabled
            if self.save_detections and detections:
                self.save_detection_result(cv_image, detections)
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def process_detections(self, results, image):
        """Process YOLO detection results"""
        detections = []
        image_height, image_width = image.shape[:2]
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x1, y1, x2, y2 = box
                
                # Get class name
                class_name = self.class_names[class_id]
                
                # Calculate center point and dimensions
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                detection = {
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'center_x': center_x, 'center_y': center_y,
                        'width': width, 'height': height
                    },
                    'normalized_bbox': {
                        'center_x': center_x / image_width,
                        'center_y': center_y / image_height,
                        'width': width / image_width,
                        'height': height / image_height
                    }
                }
                
                detections.append(detection)
        
        return detections
    
    def publish_detections(self, detections, timestamp):
        """Publish detection results as JSON string"""
        detection_data = {
            'timestamp': timestamp.sec + timestamp.nanosec * 1e-9,
            'num_detections': len(detections),
            'detections': detections
        }
        
        # Publish JSON string
        json_msg = String()
        json_msg.data = json.dumps(detection_data, indent=2)
        self.detection_publisher.publish(json_msg)
        
        # Publish bounding boxes as Float32MultiArray
        if detections:
            bbox_msg = Float32MultiArray()
            
            # Set dimensions: [num_detections, 6] (x1, y1, x2, y2, confidence, class_id)
            bbox_msg.layout.dim.append(MultiArrayDimension())
            bbox_msg.layout.dim[0].label = "detections"
            bbox_msg.layout.dim[0].size = len(detections)
            bbox_msg.layout.dim[0].stride = len(detections) * 6
            
            bbox_msg.layout.dim.append(MultiArrayDimension())
            bbox_msg.layout.dim[1].label = "bbox_data"
            bbox_msg.layout.dim[1].size = 6
            bbox_msg.layout.dim[1].stride = 6
            
            # Fill data
            for detection in detections:
                bbox = detection['bbox']
                bbox_msg.data.extend([
                    float(bbox['x1']), float(bbox['y1']),
                    float(bbox['x2']), float(bbox['y2']),
                    detection['confidence'],
                    float(detection['class_id'])
                ])
            
            self.bbox_publisher.publish(bbox_msg)
        
        # Log detection summary
        if detections:
            class_counts = {}
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            summary = ", ".join([f"{name}:{count}" for name, count in class_counts.items()])
            self.get_logger().info(f"Detected: {summary}")
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 128, 0),
            (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255)
        ]
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Choose color based on class_id
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, 
                         (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), 
                         color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image,
                         (bbox['x1'], bbox['y1'] - label_height - 5),
                         (bbox['x1'] + label_width, bbox['y1']),
                         color, -1)
            
            # Draw label text
            cv2.putText(image, label,
                       (bbox['x1'], bbox['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary info
        if detections:
            info_text = f"Detected: {len(detections)} objects"
            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def publish_annotated_image(self, annotated_image, header):
        """Publish annotated image"""
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_msg.header = header
            self.annotated_image_publisher.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing annotated image: {e}")
    
    def save_detection_result(self, image, detections):
        """Save detection result to file"""
        try:
            self.detection_count += 1
            timestamp = self.get_clock().now().nanoseconds
            
            # Save annotated image
            annotated_image = self.draw_detections(image.copy(), detections)
            image_filename = self.output_dir / f"detection_{self.detection_count:04d}_{timestamp}.jpg"
            cv2.imwrite(str(image_filename), annotated_image)
            
            # Save detection data
            json_filename = self.output_dir / f"detection_{self.detection_count:04d}_{timestamp}.json"
            detection_data = {
                'timestamp': timestamp,
                'image_file': str(image_filename),
                'detections': detections
            }
            
            with open(json_filename, 'w') as f:
                json.dump(detection_data, f, indent=2)
            
            self.get_logger().info(f"Saved detection #{self.detection_count}")
            
        except Exception as e:
            self.get_logger().error(f"Error saving detection: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        detector = CustomYOLODetector()
        
        print("🤖 Custom YOLO Detector Started!")
        print("📹 Waiting for camera images...")
        print("📡 Publishing to:")
        print("   - /custom_yolo/detections (JSON)")
        print("   - /custom_yolo/bounding_boxes (Float32MultiArray)")
        print("   - /custom_yolo/annotated_image (Image)")
        print("\n🛑 Press Ctrl+C to stop")
        
        rclpy.spin(detector)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Custom YOLO Detector...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'detector' in locals():
            detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 