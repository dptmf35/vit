#!/usr/bin/env python3

import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path

class InteractiveAnnotationEditor:
    def __init__(self, image_path, label_path, class_names):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.class_names = class_names
        
        # Load original image
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.img_height, self.img_width = self.original_image.shape[:2]
        self.display_image = self.original_image.copy()
        
        # Annotation data
        self.annotations = self.load_annotations()
        self.selected_annotation_idx = -1
        
        # Mouse interaction state
        self.drawing = False
        self.start_point = None
        self.end_point = None
        
        # Colors for different classes (BGR format)
        self.colors = self.generate_colors(len(class_names))
        
        # Window setup
        self.window_name = f"Interactive Editor - {self.image_path.name}"
        self.setup_window()
        
    def generate_colors(self, num_classes):
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def load_annotations(self):
        """Load YOLO format annotations"""
        annotations = []
        
        if not self.label_path.exists():
            return annotations
        
        with open(self.label_path, 'r') as f:
            for line_num, line in enumerate(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1 = int((center_x - width/2) * self.img_width)
                    y1 = int((center_y - height/2) * self.img_height)
                    x2 = int((center_x + width/2) * self.img_width)
                    y2 = int((center_y + height/2) * self.img_height)
                    
                    annotations.append({
                        'class_id': class_id,
                        'bbox': [x1, y1, x2, y2],
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    })
                    
                except ValueError as e:
                    print(f"Error parsing line {line_num + 1}: {line} - {e}")
                    continue
        
        return annotations
    
    def save_annotations(self):
        """Save annotations in YOLO format"""
        try:
            # Ensure label directory exists
            self.label_path.parent.mkdir(exist_ok=True)
            
            with open(self.label_path, 'w') as f:
                for ann in self.annotations:
                    class_id = ann['class_id']
                    x1, y1, x2, y2 = ann['bbox']
                    
                    # Convert to normalized YOLO format
                    center_x = (x1 + x2) / 2.0 / self.img_width
                    center_y = (y1 + y2) / 2.0 / self.img_height
                    width = (x2 - x1) / self.img_width
                    height = (y2 - y1) / self.img_height
                    
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            print(f"Saved {len(self.annotations)} annotations to: {self.label_path}")
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False
    
    def draw_annotations(self):
        """Draw all annotations on the image"""
        self.display_image = self.original_image.copy()
        
        for i, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann['bbox']
            class_id = ann['class_id']
            class_name = ann['class_name']
            
            # Use different color/thickness if selected
            if i == self.selected_annotation_idx:
                color = (0, 0, 255)  # Red for selected
                thickness = 3
            else:
                color = self.colors[class_id % len(self.colors)]
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw class label with background
            label = f"{class_id}: {class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(self.display_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), 
                         color, -1)
            
            # Label text
            cv2.putText(self.display_image, label, 
                       (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing new bbox
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.selected_annotation_idx = -1
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.drawing = False
                self.end_point = (x, y)
                
                # Create new annotation if bbox is large enough
                if (abs(self.end_point[0] - self.start_point[0]) > 20 and 
                    abs(self.end_point[1] - self.start_point[1]) > 20):
                    self.create_new_annotation()
                
                self.start_point = None
                self.end_point = None
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to select annotation
            self.select_annotation_at_point(x, y)
    
    def create_new_annotation(self):
        """Create new annotation from drawn bbox"""
        if not self.start_point or not self.end_point:
            return
        
        # Ensure proper bbox format (x1 < x2, y1 < y2)
        x1 = min(self.start_point[0], self.end_point[0])
        y1 = min(self.start_point[1], self.end_point[1])
        x2 = max(self.start_point[0], self.end_point[0])
        y2 = max(self.start_point[1], self.end_point[1])
        
        # Clamp to image bounds
        x1 = max(0, min(x1, self.img_width - 1))
        y1 = max(0, min(y1, self.img_height - 1))
        x2 = max(0, min(x2, self.img_width - 1))
        y2 = max(0, min(y2, self.img_height - 1))
        
        # Ask user for class selection
        class_id = self.select_class_simple()
        if class_id is not None:
            new_annotation = {
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2],
                'class_name': self.class_names[class_id]
            }
            self.annotations.append(new_annotation)
            print(f"Added: {self.class_names[class_id]} at [{x1}, {y1}, {x2}, {y2}]")
    
    def select_annotation_at_point(self, x, y):
        """Select annotation at the given point"""
        for i, ann in enumerate(self.annotations):
            x1, y1, x2, y2 = ann['bbox']
            if x1 <= x <= x2 and y1 <= y <= y2:
                if self.selected_annotation_idx == i:
                    # Already selected, show edit options
                    self.edit_selected_annotation()
                else:
                    self.selected_annotation_idx = i
                    print(f"Selected: {ann['class_name']} (Right-click again to edit)")
                return
        
        # No annotation found, deselect
        self.selected_annotation_idx = -1
        print("Deselected")
    
    def select_class_simple(self):
        """Simple command line class selection"""
        print("\nAvailable classes:")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        
        while True:
            try:
                choice = input(f"Select class (0-{len(self.class_names)-1}): ").strip()
                if choice == '':
                    return None
                
                class_id = int(choice)
                if 0 <= class_id < len(self.class_names):
                    return class_id
                else:
                    print(f"Invalid choice. Enter 0-{len(self.class_names)-1}")
            except ValueError:
                print("Please enter a number")
            except KeyboardInterrupt:
                return None
    
    def edit_selected_annotation(self):
        """Edit selected annotation"""
        if self.selected_annotation_idx < 0:
            return
        
        ann = self.annotations[self.selected_annotation_idx]
        print(f"\nEditing: {ann['class_name']}")
        print("1. Change class")
        print("2. Delete annotation")
        print("3. Cancel")
        
        try:
            choice = input("Choose option (1-3): ").strip()
            
            if choice == '1':
                new_class_id = self.select_class_simple()
                if new_class_id is not None:
                    self.annotations[self.selected_annotation_idx]['class_id'] = new_class_id
                    self.annotations[self.selected_annotation_idx]['class_name'] = self.class_names[new_class_id]
                    print(f"Changed to: {self.class_names[new_class_id]}")
                    
            elif choice == '2':
                confirm = input("Delete annotation? (y/N): ").strip().lower()
                if confirm == 'y':
                    del self.annotations[self.selected_annotation_idx]
                    self.selected_annotation_idx = -1
                    print("Deleted")
                    
        except KeyboardInterrupt:
            pass
    
    def setup_window(self):
        """Setup OpenCV window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Resize window if image is too large
        max_size = 1000
        if self.img_width > max_size or self.img_height > max_size:
            scale = min(max_size / self.img_width, max_size / self.img_height)
            new_width = int(self.img_width * scale)
            new_height = int(self.img_height * scale)
            cv2.resizeWindow(self.window_name, new_width, new_height)
    
    def run(self):
        """Main interaction loop"""
        print("=== Interactive Annotation Editor ===")
        print("Mouse Controls:")
        print("  Left Click + Drag: Draw new bounding box")
        print("  Right Click: Select annotation")
        print("  Right Click (on selected): Edit/Delete")
        print("\nKeyboard Controls:")
        print("  's': Save annotations")
        print("  'r': Reset to original")
        print("  'd': Delete selected annotation")
        print("  'ESC': Save and exit")
        print("  'q': Quit without saving")
        print("=" * 40)
        
        while True:
            # Draw annotations
            self.draw_annotations()
            
            # Draw current bbox being drawn
            display = self.display_image.copy()
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # Add status text
            status_text = f"Annotations: {len(self.annotations)}"
            if self.selected_annotation_idx >= 0:
                ann = self.annotations[self.selected_annotation_idx]
                status_text += f" | Selected: {ann['class_name']}"
            
            cv2.putText(display, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(display, "Left: Draw | Right: Select | 's': Save | ESC: Exit", 
                       (10, display.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                if self.save_annotations():
                    print("Saved and exiting...")
                break
                
            elif key == ord('q'):
                print("Exiting without saving...")
                break
                
            elif key == ord('s'):
                if self.save_annotations():
                    print("Annotations saved!")
                    
            elif key == ord('r'):
                self.annotations = self.load_annotations()
                self.selected_annotation_idx = -1
                print("Reset to original annotations")
                
            elif key == ord('d'):
                if self.selected_annotation_idx >= 0:
                    ann = self.annotations[self.selected_annotation_idx]
                    confirm = input(f"Delete {ann['class_name']}? (y/N): ").strip().lower()
                    if confirm == 'y':
                        del self.annotations[self.selected_annotation_idx]
                        self.selected_annotation_idx = -1
                        print("Deleted")
        
        cv2.destroyAllWindows()

def main():
    """Test the interactive editor"""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python interactive_annotation_editor.py <image_path> <label_path>")
        return
    
    # Default class names
    class_names = ["bed", "cabinet", "carpet", "chair", "closet", "countertop", "curtain", "desk", "door", "fridge",
                   "gas stove", "hanger", "kitchen cart", "lamp", "nightstand", "plant", "shelf", "sofa", 
                   "table", "tv", "window", "vanity"]
    
    try:
        editor = InteractiveAnnotationEditor(sys.argv[1], sys.argv[2], class_names)
        editor.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 