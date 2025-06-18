#!/usr/bin/env python3

import cv2
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import argparse
from pathlib import Path

class DatasetReviewer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.labels_dir = self.dataset_path / "labels"
        self.visualizations_dir = self.dataset_path / "visualizations"
        
        # Load class names from dataset.yaml
        self.class_names = self.load_class_names()
        
        # Get all image files
        self.image_files = self.get_image_files()
        self.current_index = 0
        
        # Statistics
        self.deleted_count = 0
        self.edited_count = 0
        
        # Setup GUI
        self.setup_gui()
        
        # Load first image
        if self.image_files:
            self.load_current_image()
    
    def load_class_names(self):
        """Load class names from dataset.yaml"""
        yaml_path = self.dataset_path / "dataset.yaml"
        class_names = []
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                content = f.read()
                # Parse names list from yaml
                import re
                names_match = re.search(r'names:\s*\[(.*?)\]', content, re.DOTALL)
                if names_match:
                    names_str = names_match.group(1)
                    # Extract class names
                    names = re.findall(r"'([^']*)'|\"([^\"]*)\"", names_str)
                    class_names = [name[0] or name[1] for name in names]
        
        if not class_names:
            # Default class names
            class_names = ["air purifier", "bed", "cabinet", "carpet", "chair", "closet", "countertop", "curtain", "desk", "door", "fridge",
                          "gas stove", "hanger", "kitchen cart", "lamp", "nightstand", "plant", "shelf", "sofa", 
                          "table", "tv", "window", "vanity"]
        
        print(f"Loaded {len(class_names)} classes: {class_names}")
        return class_names
    
    def get_image_files(self):
        """Get all image files in the images directory"""
        if not self.images_dir.exists():
            print(f"Images directory not found: {self.images_dir}")
            return []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(str(self.images_dir / ext)))
        
        image_files.sort()
        print(f"Found {len(image_files)} images")
        return image_files
    
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("YOLO Dataset Reviewer")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 12))
        self.info_label.pack(side=tk.LEFT)
        
        self.stats_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.stats_label.pack(side=tk.RIGHT)
        
        # Image frame
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg='gray')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(fill=tk.X, pady=(0, 10))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Annotations frame
        annotations_frame = ttk.LabelFrame(main_frame, text="Annotations", padding=10)
        annotations_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Annotations listbox
        self.annotations_listbox = tk.Listbox(annotations_frame, height=6)
        self.annotations_listbox.pack(fill=tk.X, pady=(0, 10))
        self.annotations_listbox.bind('<Double-Button-1>', self.edit_annotation)
        
        # Control buttons frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)
        
        # Navigation buttons
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(side=tk.LEFT)
        
        ttk.Button(nav_frame, text="◀◀ First", command=self.first_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Last ▶▶", command=self.last_image).pack(side=tk.LEFT, padx=2)
        
        # Action buttons
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(side=tk.RIGHT)
        
        ttk.Button(action_frame, text="🗑️ Delete", command=self.delete_current, 
                  style="Danger.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="✏️ Edit Text", command=self.edit_labels).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🎯 Visual Edit", command=self.visual_edit_labels).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="🔄 Refresh", command=self.refresh_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="📊 Statistics", command=self.show_statistics).pack(side=tk.LEFT, padx=2)
        
        # Configure button styles
        style = ttk.Style()
        style.configure("Danger.TButton", foreground="red")
        
        # Keyboard bindings
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
    
    def load_current_image(self):
        """Load and display the current image with annotations"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_index]
        image_name = Path(image_path).stem
        
        # Update info label
        self.info_label.config(text=f"Image {self.current_index + 1}/{len(self.image_files)}: {Path(image_path).name}")
        self.stats_label.config(text=f"Deleted: {self.deleted_count} | Edited: {self.edited_count}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # Load annotations
        label_path = self.labels_dir / f"{image_name}.txt"
        annotations = self.load_annotations(label_path, image.shape)
        
        # Draw annotations on image
        annotated_image = self.draw_annotations(image.copy(), annotations)
        
        # Convert to RGB for display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Display image
        self.display_image(annotated_image)
        
        # Update annotations listbox
        self.update_annotations_listbox(annotations)
    
    def load_annotations(self, label_path, image_shape):
        """Load YOLO format annotations"""
        annotations = []
        
        if not label_path.exists():
            return annotations
        
        img_height, img_width = image_shape[:2]
        
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    class_id = int(parts[0])
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    width = float(parts[3]) * img_width
                    height = float(parts[4]) * img_height
                    
                    # Convert to corner coordinates
                    x1 = int(center_x - width / 2)
                    y1 = int(center_y - height / 2)
                    x2 = int(center_x + width / 2)
                    y2 = int(center_y + height / 2)
                    
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    annotations.append({
                        'line_num': line_num,
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': (x1, y1, x2, y2),
                        'original_line': line
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing annotation line {line_num}: {line} - {e}")
        
        return annotations
    
    def draw_annotations(self, image, annotations):
        """Draw bounding boxes and labels on image"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        
        for i, ann in enumerate(annotations):
            x1, y1, x2, y2 = ann['bbox']
            color = colors[ann['class_id'] % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{ann['class_name']} ({ann['class_id']})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Label background
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def display_image(self, image):
        """Display image in canvas"""
        from PIL import Image, ImageTk
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Calculate display size (max 800x600 while maintaining aspect ratio)
        max_width, max_height = 800, 600
        img_width, img_height = pil_image.size
        
        scale = min(max_width / img_width, max_height / img_height, 1.0)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, new_width, new_height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
    
    def update_annotations_listbox(self, annotations):
        """Update the annotations listbox"""
        self.annotations_listbox.delete(0, tk.END)
        
        for i, ann in enumerate(annotations):
            text = f"{i+1}. {ann['class_name']} (ID: {ann['class_id']}) - {ann['bbox']}"
            self.annotations_listbox.insert(tk.END, text)
    
    def delete_current(self):
        """Delete current image and its annotation files"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        image_path = Path(self.image_files[self.current_index])
        image_name = image_path.stem
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {image_path.name} and all associated files?\n\nThis action cannot be undone!",
            icon='warning'
        )
        
        if not result:
            return
        
        # Files to delete
        files_to_delete = [
            image_path,  # Original image
            self.labels_dir / f"{image_name}.txt",  # Annotation file
            self.visualizations_dir / f"vis_{image_name}.jpg"  # Visualization
        ]
        
        deleted_files = []
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_files.append(file_path.name)
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
        
        self.deleted_count += 1
        print(f"Deleted files: {deleted_files}")
        
        # Remove from list and load next image
        self.image_files.pop(self.current_index)
        
        # Adjust current index
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files) - 1
        
        if self.image_files:
            self.load_current_image()
        else:
            messagebox.showinfo("Complete", "All images have been processed!")
            self.root.quit()
    
    def edit_labels(self):
        """Open label editing dialog"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        image_path = Path(self.image_files[self.current_index])
        image_name = image_path.stem
        label_path = self.labels_dir / f"{image_name}.txt"
        
        # Create label editor window
        self.open_label_editor(label_path)
    
    def open_label_editor(self, label_path):
        """Open a text editor for label file"""
        editor_window = tk.Toplevel(self.root)
        editor_window.title(f"Edit Labels - {label_path.name}")
        editor_window.geometry("700x500")
        editor_window.minsize(600, 400)
        
        # Read current content
        content = ""
        if label_path.exists():
            with open(label_path, 'r') as f:
                content = f.read()
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(editor_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.NONE)
        text_widget.insert(1.0, content)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        h_scroll = ttk.Scrollbar(text_frame, orient=tk.HORIZONTAL, command=text_widget.xview)
        text_widget.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Pack scrollbars first, then text widget
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Info label
        info_text = f"Format: class_id center_x center_y width height (normalized 0-1)\n"
        info_text += f"Available classes: {', '.join([f'{i}:{name}' for i, name in enumerate(self.class_names)])}"
        
        info_label = ttk.Label(editor_window, text=info_text, wraplength=580)
        info_label.pack(padx=10, pady=(0, 10))
        
        # Buttons
        button_frame = ttk.Frame(editor_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def save_changes():
            new_content = text_widget.get(1.0, tk.END).strip()
            
            # Validate format
            valid = True
            lines = new_content.split('\n') if new_content else []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    messagebox.showerror("Format Error", f"Line {line_num}: Expected 5 values, got {len(parts)}")
                    valid = False
                    break
                
                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    if class_id < 0 or class_id >= len(self.class_names):
                        messagebox.showerror("Class Error", f"Line {line_num}: Invalid class ID {class_id}")
                        valid = False
                        break
                    
                    if not all(0 <= coord <= 1 for coord in coords):
                        messagebox.showerror("Coordinate Error", f"Line {line_num}: Coordinates must be between 0 and 1")
                        valid = False
                        break
                        
                except ValueError:
                    messagebox.showerror("Format Error", f"Line {line_num}: Invalid number format")
                    valid = False
                    break
            
            if valid:
                # Ensure labels directory exists
                label_path.parent.mkdir(exist_ok=True)
                
                # Save file
                with open(label_path, 'w') as f:
                    f.write(new_content)
                
                self.edited_count += 1
                print(f"Saved changes to {label_path}")
                
                # Refresh current image
                self.refresh_current()
                editor_window.destroy()
        
        def cancel_changes():
            editor_window.destroy()
        
        save_btn = ttk.Button(button_frame, text="💾 Save", command=save_changes)
        save_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        cancel_btn = ttk.Button(button_frame, text="❌ Cancel", command=cancel_changes)
        cancel_btn.pack(side=tk.LEFT, padx=5, ipadx=10, ipady=5)
        
        # Add separator and help text
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        help_label = ttk.Label(button_frame, text="Tip: 💾 Save 버튼을 반드시 눌러야 저장됩니다!", 
                              foreground="blue", font=("Arial", 10, "bold"))
        help_label.pack(side=tk.LEFT, padx=10)
        
        # Focus on text widget
        text_widget.focus_set()
        
        # Keyboard shortcuts
        editor_window.bind('<Control-s>', lambda e: save_changes())
        editor_window.bind('<Escape>', lambda e: cancel_changes())
    
    def visual_edit_labels(self):
        """Open visual annotation editor"""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        image_path = Path(self.image_files[self.current_index])
        image_name = image_path.stem
        label_path = self.labels_dir / f"{image_name}.txt"
        
        try:
            # Import and run interactive editor
            from interactive_annotation_editor import InteractiveAnnotationEditor
            
            print(f"Opening visual editor for: {image_path.name}")
            
            # Hide main window temporarily
            self.root.withdraw()
            
            # Run interactive editor
            editor = InteractiveAnnotationEditor(str(image_path), str(label_path), self.class_names)
            editor.run()
            
            # Show main window again
            self.root.deiconify()
            
            # Refresh the display
            self.refresh_current()
            
            # Update edited count
            self.edited_count += 1
            
        except ImportError:
            messagebox.showerror("Import Error", 
                               "Cannot import interactive_annotation_editor.py\n"
                               "Make sure the file is in the same directory.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open visual editor:\n{str(e)}")
            # Show main window again in case of error
            self.root.deiconify()
    
    def edit_annotation(self, event):
        """Edit selected annotation"""
        selection = self.annotations_listbox.curselection()
        if not selection:
            return
        
        # For now, just open the label editor
        self.edit_labels()
    
    def refresh_current(self):
        """Refresh current image display"""
        self.load_current_image()
    
    def show_statistics(self):
        """Show dataset statistics"""
        stats_text = f"Dataset Statistics:\n\n"
        stats_text += f"Total Images: {len(self.image_files)}\n"
        stats_text += f"Current Image: {self.current_index + 1}\n"
        stats_text += f"Images Deleted: {self.deleted_count}\n"
        stats_text += f"Labels Edited: {self.edited_count}\n\n"
        stats_text += f"Available Classes ({len(self.class_names)}):\n"
        
        for i, name in enumerate(self.class_names):
            stats_text += f"  {i}: {name}\n"
        
        messagebox.showinfo("Statistics", stats_text)
    
    # Navigation methods
    def first_image(self):
        self.current_index = 0
        self.load_current_image()
    
    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def last_image(self):
        if self.image_files:
            self.current_index = len(self.image_files) - 1
            self.load_current_image()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        key = event.keysym.lower()
        
        if key == 'left' or key == 'a':
            self.previous_image()
        elif key == 'right' or key == 'd':
            self.next_image()
        elif key == 'delete':
            self.delete_current()
        elif key == 'e':
            self.edit_labels()
        elif key == 'v':
            self.visual_edit_labels()
        elif key == 'r':
            self.refresh_current()
        elif key == 'home':
            self.first_image()
        elif key == 'end':
            self.last_image()
        elif key == 'escape':
            self.root.quit()
    
    def run(self):
        """Start the GUI application"""
        if not self.image_files:
            messagebox.showwarning("No Images", "No images found in the dataset directory!")
            return
        
        print("=== YOLO Dataset Reviewer ===")
        print("Controls:")
        print("  Navigation: ← → A D Home End")
        print("  Actions: Delete=Del, Text Edit=E, Visual Edit=V, Refresh=R")
        print("  Exit: Esc")
        print("=" * 30)
        
        self.root.mainloop()

def main():
    parser = argparse.ArgumentParser(description='YOLO Dataset Reviewer and Editor')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory')
    parser.add_argument('--start-index', type=int, default=0, help='Starting image index')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path).expanduser()
    
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        return
    
    reviewer = DatasetReviewer(dataset_path)
    reviewer.current_index = args.start_index
    reviewer.run()

if __name__ == '__main__':
    main()
