#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLOE

class YOLOEROSNode(Node):
    def __init__(self):
        super().__init__('yoloe_ros_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize YOLOE model
        self.model = YOLOE("yoloe-11s-seg.pt")
        
        # Set text prompt to detect objects
        names = ["table", "fridge", "chair", "dish", "gas stove", "closet", "lamp", "curtain", "nightstand",
        "microwave", "tv", "sofa", "shelf", "window", "door", "bed", "cabinet", "plant", "computer"]
        self.model.set_classes(names, self.model.get_text_pe(names))
        
        # Subscribe to stereo image color topic
        self.image_sub = self.create_subscription(
            Image,
            '/stereo_image_color',
            self.image_callback,
            10
        )
        
        # Create publisher for detection results
        self.result_pub = self.create_publisher(
            Image,
            '/yoloe_detection_result',
            10
        )
        
        self.get_logger().info("YOLOE ROS2 node initialized and waiting for images...")
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # self.get_logger().info(f"Received image size : {cv_image.shape}")
            
            # Run YOLOE prediction
            results = self.model.predict(
                cv_image,
                conf=0.3,
                iou=0.5,
                max_det=100,
                agnostic_nms=False
            )
            
            # Process and publish results
            if len(results) > 0:
                # Get the annotated image
                annotated_image = results[0].plot()
                
                # Convert OpenCV image to ROS Image message and publish
                try:
                    result_msg = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
                    result_msg.header = msg.header  # Keep the original timestamp
                    self.result_pub.publish(result_msg)
                    self.get_logger().info("Published detection result image")
                except Exception as e:
                    self.get_logger().error(f"Failed to publish result image: {str(e)}")
                
                # Log detection results
                if results[0].boxes is not None:
                    num_detections = len(results[0].boxes)
                    self.get_logger().info(f"Detected {num_detections} objects")
                    
                    # Print detected class names and confidence scores
                    for i, box in enumerate(results[0].boxes):
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        class_name = self.model.names[class_id]
                        self.get_logger().info(f"Object {i+1}: {class_name} (confidence: {confidence:.2f})")
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    node = YOLOEROSNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLOE ROS2 node...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 