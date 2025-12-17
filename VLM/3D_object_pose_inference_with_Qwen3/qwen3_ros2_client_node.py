#!/usr/bin/env python3
"""
ROS2 Object Locator Node - Runs with system Python
Communicates with Qwen3 model server via ZMQ
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped, PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import zmq
import base64
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


class ObjectLocatorClientNode(Node):
    def __init__(self):
        super().__init__('object_locator_client_node')
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Initialize ZMQ client
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect("tcp://localhost:5555")
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/rgb',
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/depth',
            self.depth_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publisher for annotated image
        self.annotated_image_pub = self.create_publisher(
            Image,
            '/annotated_image',
            10
        )
        
        # Publisher for object world coordinates (for pick and place)
        self.coordinates_pub = self.create_publisher(
            Float64MultiArray,
            '/object_coordinates',
            10
        )
        
        # TF2 for coordinate transformation
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Camera frame name (from TF)
        self.camera_frame = 'Camera_OmniVision_OV9782_Color'
        self.robot_base_frame = 'base'
        
        # Data storage
        self.rgb_image = None
        self.depth_image = None
        self.camera_intrinsics = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        self.get_logger().info('Object Locator Client Node initialized')
        self.get_logger().info('Make sure Qwen3 model server is running!')
        
        # Start command input thread
        self.command_thread = threading.Thread(target=self.command_loop, daemon=True)
        self.command_thread.start()
        
    def rgb_callback(self, msg):
        """Callback for RGB image topic"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert RGB image: {str(e)}')
    
    def depth_callback(self, msg):
        """Callback for depth image topic"""
        try:
            # Depth images are typically in 16UC1 or 32FC1 format
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {str(e)}')
    
    def camera_info_callback(self, msg):
        """Callback for camera info topic"""
        if self.camera_intrinsics is None:
            self.get_logger().info('Camera intrinsics received!')
            self.get_logger().info(f'  fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}')
            self.get_logger().info(f'  cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}')
        
        self.camera_intrinsics = msg
        # Extract intrinsic parameters
        # K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        
    def pixel_to_world(self, x, y, depth_value):
        """Convert pixel coordinates to world coordinates using depth and intrinsics"""
        if self.fx is None or self.fy is None:
            return None
        
        # Check if depth value is valid
        if depth_value == 0:
            return None
        
        # Depth value is already in meters (from RealSense depth topic)
        z = float(depth_value)
        
        # Calculate world coordinates using pinhole camera model
        x_world = (x - self.cx) * z / self.fx
        y_world = (y - self.cy) * z / self.fy
        z_world = z
        
        return (x_world, y_world, z_world)
    
    def locate_object(self, object_name):
        """Send request to model server to locate object"""
        if self.rgb_image is None:
            self.get_logger().warn('No RGB image available')
            return None
        
        try:
            # Encode image for transmission
            image_bytes = self.rgb_image.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare request
            request = {
                'image': image_base64,
                'shape': self.rgb_image.shape,
                'object_name': object_name
            }
            
            # Send request
            self.get_logger().info(f'Sending request to model server for: {object_name}')
            self.zmq_socket.send_json(request)
            
            # Wait for response
            response = self.zmq_socket.recv_json()
            
            if response['success']:
                return response['coordinates']
            else:
                self.get_logger().error(f"Model server error: {response.get('error', 'Unknown error')}")
                return None
                
        except zmq.Again:
            self.get_logger().error('Request timeout - is model server running?')
            return None
        except Exception as e:
            self.get_logger().error(f'Error communicating with model server: {str(e)}')
            return None
    
    def process_command(self, object_name):
        """Process command to locate object and publish result"""
        if self.rgb_image is None or self.depth_image is None:
            self.get_logger().warn('Waiting for RGB and Depth images...')
            return
        
        if self.camera_intrinsics is None:
            self.get_logger().warn('Waiting for camera info...')
            return
        
        # Locate object using model server
        point_norm = self.locate_object(object_name)
        if point_norm is None:
            return
        
        # Convert normalized coordinates to pixel coordinates
        h, w, _ = self.rgb_image.shape
        center_x = int(point_norm[0] * w)
        center_y = int(point_norm[1] * h)
        
        self.get_logger().info(f'Object center in pixels: ({center_x}, {center_y})')
        
        # Get depth at center point
        if center_y >= self.depth_image.shape[0] or center_x >= self.depth_image.shape[1]:
            self.get_logger().error('Center point out of depth image bounds')
            return
        
        depth_value = self.depth_image[center_y, center_x]
        self.get_logger().info(f'Depth value: {depth_value}')
        
        # Convert to world coordinates
        world_coords = self.pixel_to_world(center_x, center_y, depth_value)
        if world_coords is None:
            self.get_logger().error('Failed to convert to world coordinates')
            return
        
        x_world, y_world, z_world = world_coords
        self.get_logger().info(f'Coordinates (camera frame): X={x_world:.3f}m, Y={y_world:.3f}m, Z={z_world:.3f}m')
        
        # Transform coordinates from camera frame to robot base frame
        try:
            # Create point in camera frame
            point_camera = PointStamped()
            point_camera.header.frame_id = self.camera_frame
            point_camera.header.stamp = self.get_clock().now().to_msg()
            point_camera.point.x = float(x_world)
            point_camera.point.y = float(y_world)
            point_camera.point.z = float(z_world)
            
            # Transform to robot base frame
            transform = self.tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            point_base = tf2_geometry_msgs.do_transform_point(point_camera, transform)
            
            x_robot = point_base.point.x
            y_robot = point_base.point.y
            z_robot = point_base.point.z
            
            self.get_logger().info(f'Transformed to robot base frame: X={x_robot:.3f}m, Y={y_robot:.3f}m, Z={z_robot:.3f}m')
            
            # Publish robot base frame coordinates for pick and place
            coords_msg = Float64MultiArray()
            coords_msg.data = [float(x_robot), float(y_robot), float(z_robot)]
            self.coordinates_pub.publish(coords_msg)
            self.get_logger().info(f'Published robot coordinates to /object_coordinates')
            
        except Exception as e:
            self.get_logger().error(f'TF transform failed: {str(e)}')
            self.get_logger().warn('Publishing camera frame coordinates without transform')
            # Fallback: publish camera coordinates
            coords_msg = Float64MultiArray()
            coords_msg.data = [float(x_world), float(y_world), float(z_world)]
            self.coordinates_pub.publish(coords_msg)
        
        # Create annotated image
        annotated_image = self.rgb_image.copy()
        
        # Draw circle at center
        cv2.circle(annotated_image, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.circle(annotated_image, (center_x, center_y), 10, (0, 0, 255), 2)
        
        # Draw world coordinates text
        coord_text = f"X:{x_world:.3f} Y:{y_world:.3f} Z:{z_world:.3f}m"
        
        # Calculate text position (above the center point)
        text_x = center_x - 100
        text_y = center_y - 20
        
        # Ensure text is within image bounds
        if text_y < 30:
            text_y = center_y + 30
        if text_x < 0:
            text_x = 10
        
        # Draw background rectangle for text
        (text_w, text_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_image, (text_x - 5, text_y - text_h - 5), 
                     (text_x + text_w + 5, text_y + 5), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(annotated_image, coord_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw object name
        obj_text = f"Object: {object_name}"
        cv2.putText(annotated_image, obj_text, (text_x, text_y - text_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            self.annotated_image_pub.publish(annotated_msg)
            self.get_logger().info('Published annotated image')
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {str(e)}')
    
    def command_loop(self):
        """Loop to accept commands from user"""
        print("\n========================================")
        print("Object Locator Node - Command Interface")
        print("========================================")
        print("Enter object name to locate (e.g., 'bed', 'chair', 'table')")
        print("Type 'quit' to exit")
        print("========================================\n")
        
        while rclpy.ok():
            try:
                command = input("Enter command: ").strip()
                
                if command.lower() == 'quit':
                    self.get_logger().info('Shutting down...')
                    break
                
                if command:
                    self.process_command(command)
                    
            except EOFError:
                break
            except Exception as e:
                self.get_logger().error(f'Error processing command: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    node = ObjectLocatorClientNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

