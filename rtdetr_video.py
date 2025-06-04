"""
RT-DETR Video Object Detection - Compact Version
Using Ultralytics RT-DETR model
Real-time GUI display
"""

import cv2
from ultralytics import RTDETR
import os

class RTDETRVideoDetector:
    def __init__(self, model_name="rtdetr-l.pt", conf_threshold=0.7):
        print(f"Loading RT-DETR model: {model_name}")
        self.model = RTDETR(model_name)
        self.conf_threshold = conf_threshold
        print("RT-DETR model loaded successfully!")
    
    def detect_and_draw(self, frame):
        # Run RT-DETR inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        return annotated_frame

def main():
    # Check display environment
    if 'DISPLAY' not in os.environ:
        print("Warning: DISPLAY environment variable not set")
        print("Setting DISPLAY=:0")
        os.environ['DISPLAY'] = ':0'
    
    # Initialize RT-DETR detector
    detector = RTDETRVideoDetector(model_name="rtdetr-l.pt", conf_threshold=0.7)
    
    # Open input video
    cap = cv2.VideoCapture('sample.mp4')
    if not cap.isOpened():
        print("Error: Cannot open sample.mp4")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print("Starting RT-DETR video detection...")
    print("Press 'q' to quit, 'space' to pause/resume")
    
    # Create window
    window_name = 'RT-DETR Video Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    frame_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Video ended!")
                    break
                
                frame_count += 1
                
                # RT-DETR detection and annotation
                output_frame = detector.detect_and_draw(frame)
                
                # Add frame info
                progress = (frame_count / total_frames) * 100
                info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)"
                cv2.putText(output_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Add controls info
                cv2.putText(output_frame, "Press 'q' to quit, 'space' to pause", 
                           (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(window_name, output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord(' '):  # spacebar
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s'):  # save current frame
                save_path = f"rtdetr_frame_{frame_count}.jpg"
                cv2.imwrite(save_path, output_frame)
                print(f"Frame saved: {save_path}")
    
    except Exception as e:
        print(f"Display error: {e}")
        print("Trying alternative display method...")
        
        # Fallback: save frames as images for manual viewing
        print("Saving frames as images instead...")
        os.makedirs("rtdetr_frames", exist_ok=True)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # Save every 30th frame
                output_frame = detector.detect_and_draw(frame)
                cv2.imwrite(f"rtdetr_frames/frame_{frame_count:05d}.jpg", output_frame)
                print(f"Saved frame {frame_count}")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main() 