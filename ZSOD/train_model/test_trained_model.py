#!/usr/bin/env python3

import cv2
import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np

class ModelTester:
    def __init__(self, model_path, class_names=None):
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Load class names
        if class_names:
            self.class_names = class_names
        else:
            # Try to get from model
            try:
                self.class_names = self.model.names
                if isinstance(self.class_names, dict):
                    self.class_names = list(self.class_names.values())
            except:
                # Default classes
                self.class_names = ["bed", "cabinet", "carpet", "chair", "closet", "curtain", "desk", "door", "fridge",
                                  "gas stove", "hanger", "lamp", "microwave", "nightstand", "plant", "shelf", "sofa", 
                                  "table", "tv", "window", "vanity"]
        
        print(f"Classes ({len(self.class_names)}): {self.class_names}")
        
    def predict_image(self, image_path, conf_threshold=0.25, save_result=True):
        """Predict objects in a single image"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return None
        
        print(f"\n🔍 Processing: {image_path.name}")
        
        # Run prediction
        results = self.model.predict(
            str(image_path),
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        if not results:
            print("❌ No results returned")
            return None
        
        result = results[0]
        
        # Load original image
        image = cv2.imread(str(image_path))
        annotated_image = image.copy()
        
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"✅ Found {len(result.boxes)} detections:")
            
            for i, box in enumerate(result.boxes):
                # Extract detection info
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                
                print(f"  {i+1}. {class_name}: {conf:.3f}")
                
                detections.append({
                    'class_id': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': xyxy
                })
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Choose color based on class
                color = self.get_class_color(cls)
                
                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.3f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Label background
                cv2.rectangle(annotated_image, 
                             (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0] + 10, y1), 
                             color, -1)
                
                # Label text
                cv2.putText(annotated_image, label, 
                           (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print("❌ No objects detected")
        
        # Save result if requested
        if save_result:
            output_path = image_path.parent / f"result_{image_path.name}"
            cv2.imwrite(str(output_path), annotated_image)
            print(f"💾 Result saved: {output_path}")
        
        return {
            'image_path': image_path,
            'detections': detections,
            'annotated_image': annotated_image,
            'output_path': output_path if save_result else None
        }
    
    def get_class_color(self, class_id):
        """Generate color for class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 165, 0),
            (255, 20, 147), (0, 191, 255), (255, 69, 0), (50, 205, 50), (138, 43, 226)
        ]
        return colors[class_id % len(colors)]
    
    def test_on_directory(self, input_dir, conf_threshold=0.25, save_results=True):
        """Test model on all images in a directory"""
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            print(f"❌ Directory not found: {input_dir}")
            return
        
        # Find image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(list(input_dir.glob(ext)))
        
        if not image_files:
            print(f"❌ No images found in: {input_dir}")
            return
        
        print(f"🔍 Testing on {len(image_files)} images...")
        
        results = []
        for img_path in sorted(image_files):
            result = self.predict_image(img_path, conf_threshold, save_results)
            if result:
                results.append(result)
        
        # Summary
        total_detections = sum(len(r['detections']) for r in results)
        images_with_detections = sum(1 for r in results if r['detections'])
        
        print(f"\n📊 Summary:")
        print(f"  Images processed: {len(results)}")
        print(f"  Images with detections: {images_with_detections}")
        print(f"  Total detections: {total_detections}")
        print(f"  Avg detections/image: {total_detections/len(results):.1f}")
        
        return results
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🎮 Interactive Testing Mode")
        print("Enter image path or 'q' to quit")
        
        while True:
            try:
                user_input = input("\nImage path: ").strip()
                
                if user_input.lower() == 'q':
                    break
                
                if not user_input:
                    continue
                
                # Test image
                result = self.predict_image(user_input)
                
                if result and result['detections']:
                    # Show image
                    print("Press any key to continue or 'q' to quit...")
                    cv2.imshow('Detection Result', result['annotated_image'])
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()
                    
                    if key == ord('q'):
                        break
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Interactive mode ended")

def main():
    parser = argparse.ArgumentParser(description='Test trained YOLO model')
    parser.add_argument('model_path', type=str, help='Path to trained model (.pt file)')
    parser.add_argument('--input', type=str, help='Input image or directory path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save result images')
    parser.add_argument('--interactive', action='store_true', help='Interactive testing mode')
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = ModelTester(args.model_path)
        
        if args.interactive:
            # Interactive mode
            tester.interactive_test()
            
        elif args.input:
            input_path = Path(args.input)
            
            if input_path.is_file():
                # Single image
                result = tester.predict_image(
                    args.input, 
                    conf_threshold=args.conf, 
                    save_result=not args.no_save
                )
                
                if result and result['detections']:
                    print("Press any key to close...")
                    cv2.imshow('Detection Result', result['annotated_image'])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
            elif input_path.is_dir():
                # Directory
                tester.test_on_directory(
                    args.input, 
                    conf_threshold=args.conf, 
                    save_results=not args.no_save
                )
            else:
                print(f"❌ Invalid input path: {args.input}")
        else:
            print("❌ No input specified. Use --input or --interactive")
            print("\nExamples:")
            print(f"  python3 {sys.argv[0]} model.pt --input image.jpg")
            print(f"  python3 {sys.argv[0]} model.pt --input /path/to/images/")
            print(f"  python3 {sys.argv[0]} model.pt --interactive")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main() 