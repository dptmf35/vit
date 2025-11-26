import os
import sys
import cv2
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
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs  # Required for PointStamped transform support

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint")

# Add current directory to Python path to make utils importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

"""
Hyperparam for Real-time Detection and Segmentation
"""
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"


# available models : sam2.1_hiera_large, sam2.1_hiera_base_plus, sam2.1_hiera_small, sam2.1_hiera_tiny

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

BOX_THRESHOLD = 0.50
TEXT_THRESHOLD = 0.50
# Updated object list as requested
TEXT_PROMPT = "table. chair. shelf. plant. bed. desk. door."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Real-time processing settings
PROCESS_EVERY_N_FRAMES = 3  # Process every N frames for better performance
WINDOW_NAME = "Grounded SAM2 ROS2 Real-time Demo"
CAMERA_TOPIC = "/stereo_image_color"

class GroundedSAM2ROS2Node(Node):
    """ROS2 Node for real-time Grounded SAM2 processing"""
    
    def __init__(self):
        super().__init__('grounded_sam2_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize models
        self.initialize_models()
        
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
        
        # Detection results cache (boxes, labels, masks, scores, poses)
        self.last_detection_results = ([], [], [], [], [])
        self.processing_lock = threading.Lock()
        
        # TF2 setup for coordinate transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_frame_id = None
        self.target_world_frame = "base_link"
        
        # Camera intrinsic parameters
        self.K = np.array([
            [634.0862399675711, 0.0, 640.0],
            [0.0, 634.0862399675711, 360.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Depth processing parameters
        self.depth_scale_factor = 1000.0
        self.depth_min_range = 0.05
        self.depth_max_range = 50.0
        
        # Create display window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        
        # Log initialization info
        self.get_logger().info(f"Grounded SAM2 ROS2 Node initialized")
        self.get_logger().info(f"RGB Topic: {CAMERA_TOPIC}, Depth Topic: /depth")
        self.get_logger().info(f"Target Frame: {self.target_world_frame}")
        self.get_logger().info(f"Text Prompt: '{TEXT_PROMPT}'")
        self.get_logger().info(f"Device: {DEVICE}")
        
    def initialize_models(self):
        """Initialize Grounding DINO and SAM2 models"""
        self.get_logger().info("Loading models...")
        
        # Build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )
        
        # Build SAM2 image predictor
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        self.get_logger().info("Models loaded successfully!")
    
    def image_callback(self, msg):
        """Callback function for image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Track camera frame_id from message
            self.camera_frame_id = msg.header.frame_id if msg.header.frame_id else self.camera_frame_id
            
            # Update current frame
            self.current_frame = cv_image
            
            # Update frame count and FPS
            self.frame_count += 1
            self.fps_counter += 1
            
            # Calculate FPS every 10 frames
            if self.fps_counter >= 10:
                current_time = time.time()
                self.fps_display = self.fps_counter / (current_time - self.frame_start_time)
                self.frame_start_time = current_time
                self.fps_counter = 0
            
            # Process frame for detection/segmentation
            if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
                self.get_logger().info(f"Processing frame {self.frame_count}...")
                self.process_current_frame()
            
            # Update display
            self.update_display()
            
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")
    
    def depth_callback(self, msg):
        """Callback function for depth messages"""
        try:
            with self.depth_lock:
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                if len(self.latest_depth_image.shape) == 3:
                    self.latest_depth_image = self.latest_depth_image[:, :, 0]
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")
    
    def process_current_frame(self):
        """Process current frame with Grounding DINO + SAM2"""
        if self.current_frame is None:
            return
            
        try:
            with self.processing_lock:
                frame = self.current_frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Save to /tmp (usually tmpfs, so fast)
                temp_path = "/tmp/grounded_sam2_frame.jpg"
                pil_image.save(temp_path, quality=95, optimize=False)
                
                # Run Grounding DINO
                image_source, image = load_image(temp_path)
                
                with torch.cuda.amp.autocast(enabled=False):
                    boxes, confidences, labels = predict(
                        model=self.grounding_model,
                        image=image,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD,
                    )
                
                if len(boxes) == 0:
                    self.last_detection_results = ([], [], [], [], [])
                    return
                
                # Convert boxes to image coordinates
                h, w = frame.shape[:2]
                boxes_scaled = boxes * torch.tensor([w, h, w, h], device=boxes.device)
                
                # Convert from center format to corner format
                boxes_xyxy = torch.zeros_like(boxes_scaled)
                boxes_xyxy[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2  # x1
                boxes_xyxy[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2  # y1
                boxes_xyxy[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2  # x2
                boxes_xyxy[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2  # y2
                
                # Run SAM2 on the frame
                self.sam2_predictor.set_image(frame_rgb)
                
                # Use torch.no_grad() to avoid gradient warnings
                with torch.no_grad():
                    masks, scores, logits = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes_xyxy.cpu().numpy(),
                        multimask_output=False,
                    )
                
                # Convert tensors to numpy
                boxes_np = boxes_xyxy.cpu().numpy()
                confidences_np = confidences.cpu().numpy()
                
                # Estimate poses for each detected object
                poses = []
                with self.depth_lock:
                    current_depth = self.latest_depth_image
                
                debug_this_frame = self.frame_count <= 3
                
                # Get TF transform once for all objects (optimization)
                transform_matrix = None
                try:
                    if self.camera_frame_id:
                        transform = self.tf_buffer.lookup_transform(
                            self.target_world_frame,
                            self.camera_frame_id,
                            rclpy.time.Time(),
                            timeout=Duration(seconds=0.01)
                        )
                        # Convert to 4x4 matrix
                        trans = transform.transform.translation
                        rot = transform.transform.rotation
                        
                        # Quaternion to rotation matrix
                        x, y, z, w = rot.x, rot.y, rot.z, rot.w
                        R = np.array([
                            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
                        ])
                        transform_matrix = np.eye(4)
                        transform_matrix[:3, :3] = R
                        transform_matrix[:3, 3] = [trans.x, trans.y, trans.z]
                except Exception:
                    pass  # Will use camera frame as fallback
                
                if current_depth is not None:
                    for i, mask in enumerate(masks):
                        if mask.ndim == 3:
                            mask = mask[0]
                        
                        pose = self.estimate_pose_from_segmentation(
                            mask, current_depth, debug_this_frame, transform_matrix
                        )
                        
                        if pose is not None:
                            pose['label'] = labels[i] if i < len(labels) else f"Object_{i}"
                            pose['confidence'] = float(confidences_np[i])
                            poses.append(pose)
                            
                            self.get_logger().info(
                                f"Pose [{pose['label']}]: "
                                f"({pose['x']:.3f}, {pose['y']:.3f}), "
                                f"Dist: {pose['distance']:.3f}m"
                            )
                
                # Cache results for display
                self.last_detection_results = (boxes_np, labels, masks, confidences_np, poses)
                
        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")
            self.last_detection_results = ([], [], [], [], [])
    
    def estimate_pose_from_segmentation(self, mask, depth_image, debug=False, transform_matrix=None):
        """Estimate 3D pose from segmentation center and depth"""
        if depth_image is None:
            return None
            
        try:
            # Convert mask to boolean if needed
            mask_bool = mask > 0.5 if mask.dtype != bool else mask
            
            # Find segmentation center
            mask_indices = np.where(mask_bool)
            if len(mask_indices[0]) == 0:
                return None
                
            # Calculate center of mass
            center_y = int(np.mean(mask_indices[0]))
            center_x = int(np.mean(mask_indices[1]))
            
            # Snap to nearest in-mask pixel if center is outside
            if not mask_bool[center_y, center_x]:
                mask_y, mask_x = mask_indices[0], mask_indices[1]
                distances_sq = (mask_y - center_y)**2 + (mask_x - center_x)**2
                nearest_idx = int(np.argmin(distances_sq))
                center_y = int(mask_y[nearest_idx])
                center_x = int(mask_x[nearest_idx])
            
            # Extract and validate depth values
            mask_depths = depth_image[mask_bool]
            valid_depths = mask_depths[(mask_depths > 0) & (mask_depths < 65535) & np.isfinite(mask_depths)]
            
            if len(valid_depths) == 0:
                return None
            
            # Try multiple scale factors
            clipped_depths = None
            for scale_factor in [self.depth_scale_factor, 1.0, 100.0, 10000.0]:
                depths_m = valid_depths / scale_factor
                clipped = depths_m[(depths_m >= self.depth_min_range) & (depths_m <= self.depth_max_range)]
                if len(clipped) > 0:
                    clipped_depths = clipped
                    break
            
            # Fallback with wider range
            if clipped_depths is None or len(clipped_depths) == 0:
                depths_m = valid_depths / self.depth_scale_factor
                clipped_depths = depths_m[(depths_m >= 0.01) & (depths_m <= 100.0)]
                if len(clipped_depths) == 0:
                    return None
            
            depth_meters = float(np.median(clipped_depths))
            
            if not (self.depth_min_range <= depth_meters <= self.depth_max_range and np.isfinite(depth_meters)):
                return None
            
            # Convert pixel to camera coordinates
            x_cam = (center_x - self.K[0, 2]) * depth_meters / self.K[0, 0]
            y_cam = (center_y - self.K[1, 2]) * depth_meters / self.K[1, 1]
            z_cam = depth_meters
            
            # Transform to base_link using pre-computed transform matrix
            x_world, y_world, z_world = x_cam, y_cam, z_cam
            if transform_matrix is not None:
                point_homo = np.array([x_cam, y_cam, z_cam, 1.0])
                transformed = transform_matrix @ point_homo
                x_world, y_world, z_world = transformed[0], transformed[1], transformed[2]
            
            # Calculate distance and angle
            distance = np.sqrt(x_world**2 + y_world**2)
            theta = math.atan2(y_world, x_world)
            
            pose = {
                'x': float(x_world),
                'y': float(y_world),
                'distance': float(distance),
                'theta': float(theta),
                'depth_pixels_used': int(len(clipped_depths)),
                'center_pixel': [int(center_x), int(center_y)]
            }
            
            return pose
            
        except Exception as e:
            self.get_logger().error(f"Pose estimation error: {e}")
            return None
        
    def draw_detections_on_frame(self, frame, boxes, labels, masks, scores, poses=None):
        """Draw bounding boxes, labels, masks, and poses on frame"""
        result_frame = frame.copy()
        
        if len(boxes) > 0:
            # Draw masks with colored overlay
            for i, mask in enumerate(masks):
                mask = mask[0] if mask.ndim == 3 else mask
                mask = mask.astype(bool)
                
                np.random.seed(i)
                color = np.random.randint(50, 255, 3, dtype=np.uint8)
                
                result_frame[mask] = cv2.addWeighted(
                    result_frame[mask], 0.6,
                    np.full_like(result_frame[mask], color), 0.4, 0
                )
            
            # Draw bounding boxes, labels, and poses
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label text
                label_text = str(labels[i]) if labels and i < len(labels) else f"Object {i+1}"
                if scores is not None and i < len(scores):
                    label_text += f": {scores[i]:.2f}"
                
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(result_frame, (x1, y1-th-10), (x1+tw, y1), (0, 255, 0), -1)
                cv2.putText(result_frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Draw pose if available
                if poses and i < len(poses):
                    pose = poses[i]
                    cx, cy = pose['center_pixel']
                    cv2.circle(result_frame, (cx, cy), 5, (0, 255, 255), -1)
                    
                    pose_y = y1 + 20
                    cv2.rectangle(result_frame, (x1, pose_y-15), (x1+200, pose_y+25), (0, 0, 0), -1)
                    cv2.putText(result_frame, f"Pose: ({pose['x']:.2f}, {pose['y']:.2f})", 
                              (x1+5, pose_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(result_frame, f"Dist: {pose['distance']:.2f}m", 
                              (x1+5, pose_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return result_frame
    
    def update_display(self):
        """Update display window with detections"""
        if self.current_frame is None:
            return
            
        try:
            with self.processing_lock:
                frame = self.current_frame.copy()
                boxes, labels, masks, scores, poses = self.last_detection_results
                
                result_frame = self.draw_detections_on_frame(frame, boxes, labels, masks, scores, poses) \
                              if len(boxes) > 0 else frame
                
                cv2.imshow(WINDOW_NAME, result_frame)
                
                # Log FPS
                self.get_logger().info(f"FPS: {self.fps_display:.1f} | Frame: {self.frame_count}")
                
                # Handle quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.get_logger().info("Quit key pressed, shutting down...")
                    rclpy.shutdown()
                    
        except Exception as e:
            self.get_logger().error(f"Display update error: {e}")

def main(args=None):
    """Main function to run the ROS2 node"""
    print("=== Grounded SAM2 ROS2 Real-time Demo ===")
    print(f"Device: {DEVICE}")
    print(f"Camera Topic: {CAMERA_TOPIC}")
    print(f"Text Prompt: {TEXT_PROMPT}")
    
    # Enable TF32 for better performance on Ampere+ GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    rclpy.init(args=args)
    
    try:
        node = GroundedSAM2ROS2Node()
        print("\nNode ready. Press 'q' in display window to quit.\n")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        try:
            if 'node' in locals():
                total_time = time.time() - node.start_time
                avg_fps = node.frame_count / total_time if total_time > 0 else 0
                print(f"\n{'='*50}")
                print(f"Statistics: {node.frame_count} frames | {total_time:.1f}s | {avg_fps:.1f} FPS")
                print(f"{'='*50}")
                node.destroy_node()
        except:
            pass
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == "__main__":
    main() 