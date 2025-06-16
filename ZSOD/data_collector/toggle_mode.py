#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from std_msgs.msg import String
import sys

class ModeToggler(Node):
    def __init__(self):
        super().__init__('mode_toggler')
        
        # Create service client
        self.client = self.create_client(SetBool, '/toggle_collection_mode')
        
        # Subscribe to mode status
        self.status_sub = self.create_subscription(
            String,
            '/collector_mode_status',
            self.status_callback,
            10
        )
        
        self.current_mode = None
    
    def status_callback(self, msg):
        self.current_mode = msg.data
    
    def toggle_to_collection_mode(self):
        """Switch to collection mode"""
        request = SetBool.Request()
        request.data = True
        
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service not available')
            return False
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Response: {response.message}')
            return response.success
        else:
            self.get_logger().error('Service call failed')
            return False
    
    def toggle_to_test_mode(self):
        """Switch to test mode"""
        request = SetBool.Request()
        request.data = False
        
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service not available')
            return False
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Response: {response.message}')
            return response.success
        else:
            self.get_logger().error('Service call failed')
            return False

def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 toggle_mode.py test      # Switch to test mode")
        print("  python3 toggle_mode.py collect   # Switch to collection mode")
        print("  python3 toggle_mode.py status    # Show current status")
        sys.exit(1)
    
    rclpy.init()
    
    toggler = ModeToggler()
    
    mode = sys.argv[1].lower()
    
    if mode == 'test':
        print("Switching to TEST MODE...")
        success = toggler.toggle_to_test_mode()
        if success:
            print("✅ Successfully switched to TEST MODE")
        else:
            print("❌ Failed to switch mode")
    
    elif mode == 'collect' or mode == 'collection':
        print("Switching to COLLECTION MODE...")
        success = toggler.toggle_to_collection_mode()
        if success:
            print("✅ Successfully switched to COLLECTION MODE")
        else:
            print("❌ Failed to switch mode")
    
    elif mode == 'status':
        print("Checking current mode...")
        # Wait a bit to receive status
        import time
        time.sleep(1.0)
        rclpy.spin_once(toggler, timeout_sec=1.0)
        
        if toggler.current_mode:
            if toggler.current_mode == "TEST_MODE":
                print("🔍 Current mode: TEST MODE (detection only)")
            elif toggler.current_mode == "COLLECTION_MODE":
                print("💾 Current mode: COLLECTION MODE (saving data)")
            else:
                print(f"Current mode: {toggler.current_mode}")
        else:
            print("❌ Could not retrieve current mode status")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: test, collect, status")
        sys.exit(1)
    
    toggler.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 