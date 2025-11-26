import os
import sys
import cv2
import json
import torch
import numpy as np
from PIL import Image
import time
import warnings
import threading
import math

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROS_Image
from cv_bridge import CvBridge
from rclpy.duration import Duration
import tf2_ros

# Transformers for RF-DETR
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
DATASET_PATH = "/home/yeseul/yolo_dataset_1015"
CLASS_MAP_FILE = os.path.join(DATASET_PATH, "coco_annotations", "class_map.json")
MODEL_PATH = "/home/yeseul/Desktop/mygitrepos/vit/DETR-series/rf_detr_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Real-time processing settings
PROCESS_EVERY_N_FRAMES = 1  # Process every frame if possible, or increase if slow
WINDOW_NAME = "RF-DETR ROS2 Real-time Demo"
CAMERA_TOPIC = "/stereo_image_color"
CONFIDENCE_THRESHOLD = 0.05

class RFDetrROS2Node(Node):
    """ROS2 Node for real-time RF-DETR object detection"""
    
    def __init__(self):
        super().__init__('rf_detr_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load Class Map
        self.class_map = self.load_class_map()
        
        # Initialize model
        self.initialize_model()
        
        # Create image subscriber
        self.image_subscription = self.create_subscription(
            ROS_Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )
        
        # Create depth subscriber
        self.depth_subscription = self.create_subscription(
            ROS_Image,
            "/depth",
            self.depth_callback,
            10
        )
        
        # Depth image storage
        self.latest_depth_image = None
        self.depth_lock = threading.Lock()
        
        # Frame processing variables
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_counter = 0
        self.fps_display = 0.0
        self.frame_start_time = time.time()
        self.current_frame = None
        
        # Detection results cache (boxes, labels, scores, poses)
        self.last_detection_results = ([], [], [], [])
        self.processing_lock = threading.Lock()
        
        # TF2 setup for coordinate transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_frame_id = None
        self.target_world_frame = "base_link"
        
        # Camera intrinsic parameters (approximate/default or calibrated)
        # Using same values as in the reference script
        self.K = np.array([
            [634.0862399675711, 0.0, 640.0],
            [0.0, 634.0862399675711, 360.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Depth processing parameters
        self.depth_scale_factor = 1000.0
        self.depth_min_range = 0.05
        self.depth_max_range = 50.0
        
        # Create result image publisher
        self.result_publisher = self.create_publisher(
            ROS_Image,
            "/rf_detr/detection_result",
            10
        )
        
        self.get_logger().info(f"RF-DETR ROS2 Node initialized")
        self.get_logger().info(f"Model Path: {MODEL_PATH}")
        self.get_logger().info(f"Device: {DEVICE}")

    def load_class_map(self):
        try:
            with open(CLASS_MAP_FILE, 'r') as f:
                data = json.load(f)
            # Ensure keys are strings
            return {str(k): v for k, v in data.items()}
        except Exception as e:
            self.get_logger().error(f"Failed to load class map from {CLASS_MAP_FILE}: {e}")
            return {}

    def initialize_model(self):
        """Initialize RF-DETR model"""
        self.get_logger().info("Loading RF-DETR model...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
            self.model = AutoModelForObjectDetection.from_pretrained(MODEL_PATH)
            self.model.to(DEVICE)
            self.model.eval()
            self.get_logger().info("RF-DETR model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
            sys.exit(1)
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Track camera frame_id
            self.camera_frame_id = msg.header.frame_id if msg.header.frame_id else self.camera_frame_id
            
            self.current_frame = cv_image
            self.frame_count += 1
            self.fps_counter += 1
            
            # Calculate FPS
            if self.fps_counter >= 10:
                current_time = time.time()
                self.fps_display = self.fps_counter / (current_time - self.frame_start_time)
                self.frame_start_time = current_time
                self.fps_counter = 0
            
            # Process frame
            if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
                self.process_current_frame()
            
            # Update display
            self.update_display()
            
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
    
    def depth_callback(self, msg):
        try:
            with self.depth_lock:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                if len(self.latest_depth_image.shape) == 3:
                    self.latest_depth_image = self.latest_depth_image[:, :, 0]
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def process_current_frame(self):
        if self.current_frame is None:
            return

        try:
            with self.processing_lock:
                frame = self.current_frame.copy()
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Inference
                inputs = self.processor(images=pil_image, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                target_sizes = torch.tensor([pil_image.size[::-1]])
                results = self.processor.post_process_object_detection(
                    outputs, 
                    target_sizes=target_sizes, 
                    threshold=CONFIDENCE_THRESHOLD
                )[0]
                
                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"].cpu().numpy()
                
                # Map label IDs to names
                label_names = [self.class_map.get(str(l), str(l)) for l in labels]
                
                # Estimate poses
                poses = []
                with self.depth_lock:
                    current_depth = self.latest_depth_image
                
                # Get TF transform
                transform_matrix = self.get_transform_matrix()
                
                if current_depth is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        pose = self.estimate_pose_from_box(
                            box, current_depth, transform_matrix
                        )
                        if pose is not None:
                            pose['label'] = label_names[i]
                            pose['confidence'] = float(scores[i])
                            poses.append(pose)
                
                self.last_detection_results = (boxes, label_names, scores, poses)
                
        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")
            self.last_detection_results = ([], [], [], [])

    def get_transform_matrix(self):
        try:
            if self.camera_frame_id:
                transform = self.tf_buffer.lookup_transform(
                    self.target_world_frame,
                    self.camera_frame_id,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.01)
                )
                trans = transform.transform.translation
                rot = transform.transform.rotation
                
                x, y, z, w = rot.x, rot.y, rot.z, rot.w
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                    [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                    [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
                ])
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = R
                transform_matrix[:3, 3] = [trans.x, trans.y, trans.z]
                return transform_matrix
        except Exception:
            return None
        return None

    def estimate_pose_from_box(self, box, depth_image, transform_matrix=None):
        """Estimate 3D pose from bounding box center and depth"""
        if depth_image is None:
            return None
            
        try:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure box is within image bounds
            h, w = depth_image.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x1 >= x2 or y1 >= y2:
                return None
                
            # Center of box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Extract depth region (center area or whole box)
            # Using a small window around center for depth estimation to avoid background
            window_size = min(x2-x1, y2-y1) // 4
            window_size = max(1, window_size)
            
            wx1 = max(x1, center_x - window_size)
            wx2 = min(x2, center_x + window_size)
            wy1 = max(y1, center_y - window_size)
            wy2 = min(y2, center_y + window_size)
            
            depth_crop = depth_image[wy1:wy2, wx1:wx2]
            valid_depths = depth_crop[(depth_crop > 0) & (depth_crop < 65535) & np.isfinite(depth_crop)]
            
            if len(valid_depths) == 0:
                return None
                
            # Depth scaling logic similar to previous script
            clipped_depths = None
            for scale_factor in [self.depth_scale_factor, 1.0, 100.0, 10000.0]:
                depths_m = valid_depths / scale_factor
                clipped = depths_m[(depths_m >= self.depth_min_range) & (depths_m <= self.depth_max_range)]
                if len(clipped) > 0:
                    clipped_depths = clipped
                    break
            
            if clipped_depths is None:
                return None
                
            depth_meters = float(np.median(clipped_depths))
            
            # Back-project to 3D
            x_cam = (center_x - self.K[0, 2]) * depth_meters / self.K[0, 0]
            y_cam = (center_y - self.K[1, 2]) * depth_meters / self.K[1, 1]
            z_cam = depth_meters
            
            x_world, y_world, z_world = x_cam, y_cam, z_cam
            if transform_matrix is not None:
                point_homo = np.array([x_cam, y_cam, z_cam, 1.0])
                transformed = transform_matrix @ point_homo
                x_world, y_world, z_world = transformed[0], transformed[1], transformed[2]
                
            distance = np.sqrt(x_world**2 + y_world**2)
            
            return {
                'x': float(x_world),
                'y': float(y_world),
                'distance': float(distance),
                'center_pixel': [center_x, center_y]
            }
            
        except Exception as e:
            # self.get_logger().error(f"Pose estimation error: {e}")
            return None

    def draw_detections_on_frame(self, frame, boxes, labels, scores, poses=None):
        result_frame = frame.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label_text = labels[i]
            if scores is not None and i < len(scores):
                label_text += f": {scores[i]:.2f}"
            
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_frame, (x1, y1-th-10), (x1+tw, y1), (0, 255, 0), -1)
            cv2.putText(result_frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Draw pose
            if poses:
                # Find matching pose for this box (simplification: assuming 1:1 mapping if generated sequentially)
                # Actually poses list might filter out invalid depths, so we need to match carefully.
                # Here we stored all detections, but poses list only contains valid ones.
                # Let's just iterate poses and match by proximity or index if logic aligns.
                # In process_current_frame, we iterate boxes and append to poses if valid.
                # So poses list index does NOT match boxes list index directly if some failed.
                # Better approach: Store pose in a dict or structure aligned with boxes, or find by center.
                pass
        
        # Draw poses separately
        if poses:
            for pose in poses:
                cx, cy = pose['center_pixel']
                cv2.circle(result_frame, (cx, cy), 5, (0, 255, 255), -1)
                cv2.putText(result_frame, f"({pose['x']:.2f}, {pose['y']:.2f})m", 
                          (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return result_frame

    def update_display(self):
        if self.current_frame is None:
            return
            
        try:
            with self.processing_lock:
                frame = self.current_frame.copy()
                boxes, labels, scores, poses = self.last_detection_results
                
                result_frame = self.draw_detections_on_frame(frame, boxes, labels, scores, poses)
                
                # Publish result image
                try:
                    ros_msg = self.bridge.cv2_to_imgmsg(result_frame, "bgr8")
                    # Set header
                    ros_msg.header.stamp = self.get_clock().now().to_msg()
                    if self.camera_frame_id:
                        ros_msg.header.frame_id = self.camera_frame_id
                    
                    self.result_publisher.publish(ros_msg)
                    
                    # Log FPS occasionally
                    if self.frame_count % 30 == 0:
                         self.get_logger().info(f"Publishing result... FPS: {self.fps_display:.1f}")
                         
                except Exception as e:
                    self.get_logger().error(f"Failed to convert/publish image: {e}")
                    
        except Exception as e:
            self.get_logger().error(f"Display update error: {e}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RFDetrROS2Node()
        print("\nNode ready. Publishing to /rf_detr/detection_result\n")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # cv2.destroyAllWindows() # No GUI windows to destroy
        rclpy.shutdown()

if __name__ == "__main__":
    main()

