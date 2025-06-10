"""
RT-DETR Video Object Detection - Compact Version
Using Ultralytics RT-DETR model
"""

import cv2
from ultralytics import RTDETR
import time

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
    # Initialize RT-DETR detector
    detector = RTDETRVideoDetector(model_name="rtdetr-l.pt", conf_threshold=0.7)
    
    # Open video
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
    print("Processing video with RT-DETR... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_display = 0.0
    fps_update_time = start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
        
        frame_count += 1
        fps_counter += 1
        
        # RT-DETR detection and annotation
        output_frame = detector.detect_and_draw(frame)
        
        # Calculate real-time FPS every second
        current_time = time.time()
        if current_time - fps_update_time >= 1.0:  # Update every 1 second
            fps_display = fps_counter / (current_time - fps_update_time)
            fps_counter = 0
            fps_update_time = current_time
        
        # Calculate progress
        progress = (frame_count / total_frames) * 100
        
        # Add frame counter and real-time FPS
        info_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {fps_display:.1f}"
        cv2.putText(output_frame, info_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add average processing speed
        elapsed_total = current_time - start_time
        avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
        avg_text = f"Avg FPS: {avg_fps:.1f}"
        cv2.putText(output_frame, avg_text, 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display
        cv2.imshow('RT-DETR Video Detection', output_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    final_avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Processed {frame_count} frames in {total_time:.2f}s")
    print(f"Average processing speed: {final_avg_fps:.2f} FPS")

if __name__ == "__main__":
    main() 