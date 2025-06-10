"""
DETR Video Object Detection - Compact Version
"""

import torch
import cv2
from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np

class DETRVideoDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.threshold = threshold
        print(f"DETR loaded on {self.device}")
    
    def detect_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Process with DETR
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
        
        return results
    
    def draw_boxes(self, frame, results):
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = self.model.config.id2label[label.item()]
            confidence = score.item()
            
            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def main():
    # Initialize detector
    detector = DETRVideoDetector(threshold=0.7)
    
    # Open video
    cap = cv2.VideoCapture('sample.mp4')
    if not cap.isOpened():
        print("Error: Cannot open sample.mp4")
        return
    
    print("Processing video... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
        
        # Detect objects
        results = detector.detect_frame(frame)
        
        # Draw results
        output_frame = detector.draw_boxes(frame, results)
        
        # Display
        cv2.imshow('DETR Video Detection', output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 