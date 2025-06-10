# #!/usr/bin/env python3
# """
# YOLOE Official Prompt-Free Object Detection
# Based on the official GitHub implementation

# This script follows the exact approach from THU-MIG/yoloe repository
# for proper prompt-free detection with vocabulary embedding.
# """

# import cv2
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from ultralytics import YOLOE
# from ultralytics.models.yolo.yoloe.val_pe_free import YOLOEPEFreeDetectValidator
# import torch
# import requests
# import yaml
# import os

# def download_ram_tag_list(save_path="ram_tag_list.txt"):
#     """
#     Download RAM tag list from GitHub
#     """
#     if os.path.exists(save_path):
#         print(f"✅ RAM tag list exists: {save_path}")
#         return save_path
    
#     url = "https://raw.githubusercontent.com/xinyu1205/recognize-anything/main/ram/data/ram_tag_list.txt"
    
#     try:
#         print("📥 Downloading RAM tag list...")
#         response = requests.get(url)
#         response.raise_for_status()
        
#         with open(save_path, 'w', encoding='utf-8') as f:
#             f.write(response.text)
        
#         print(f"✅ Downloaded: {save_path}")
#         return save_path
        
#     except Exception as e:
#         print(f"❌ Download failed: {e}")
#         return None

# def download_model_files():
#     """
#     Download required model files if they don't exist
#     """
#     model_files = {
#         "yoloe-v8l-seg.pt": "jameslahm/yoloe-v8l-seg",
#         "yoloe-v8l-seg-pf.pt": "jameslahm/yoloe-v8l-seg-pf"
#     }
    
#     for filename, model_name in model_files.items():
#         if not os.path.exists(filename):
#             try:
#                 print(f"📥 Downloading {filename}...")
#                 model = YOLOE.from_pretrained(model_name)
#                 # Save the model
#                 torch.save(model.model.state_dict(), filename)
#                 print(f"✅ Downloaded: {filename}")
#             except Exception as e:
#                 print(f"❌ Failed to download {filename}: {e}")

# def setup_official_prompt_free_model(image_path, device="cuda"):
#     """
#     Setup YOLOE with official prompt-free approach
#     """
#     print("🔧 Setting up YOLOE with official prompt-free method...")
    
#     # Download required files
#     ram_tag_file = download_ram_tag_list()
#     if not ram_tag_file:
#         print("❌ Failed to download RAM tag list")
#         return None, None, None
    
#     download_model_files()
    
#     try:
#         # Step 1: Load unfused model for vocabulary generation
#         print("🏗️ Loading unfused model...")
        
#         # Try different ways to load the model
#         config_file = "yoloe-v8l.yaml"
#         weight_file = "yoloe-v8l-seg.pt"
        
#         if os.path.exists(config_file):
#             unfused_model = YOLOE(config_file)
#         else:
#             print("⚠️ Config file not found, using weight file directly")
#             unfused_model = YOLOE(weight_file)
        
#         if os.path.exists(weight_file):
#             unfused_model.load(weight_file)
#         else:
#             print("⚠️ Weight file not found, trying to load pretrained...")
#             unfused_model = YOLOE.from_pretrained("jameslahm/yoloe-v8l-seg")
        
#         unfused_model.eval()
        
#         if device == "cuda" and torch.cuda.is_available():
#             unfused_model.cuda()
        
#         # Step 2: Load RAM tag list
#         print("📚 Loading RAM tag list...")
#         with open(ram_tag_file, 'r', encoding='utf-8') as f:
#             names = [x.strip() for x in f.readlines() if x.strip()]
        
#         print(f"📝 Loaded {len(names)} vocabulary terms")
#         print(f"🏷️ Sample: {names[:10]}...")
        
#         # Step 3: Generate vocabulary embeddings
#         print("🧠 Generating vocabulary embeddings...")
#         vocab = unfused_model.get_vocab(names)
        
#         # Step 4: Load prompt-free model
#         print("🎯 Loading prompt-free model...")
#         pf_weight_file = "yoloe-v8l-seg-pf.pt"
        
#         if os.path.exists(pf_weight_file):
#             model = YOLOE(pf_weight_file)
#         else:
#             print("⚠️ Prompt-free weight file not found, using regular model...")
#             model = unfused_model  # Use the same model
        
#         if device == "cuda" and torch.cuda.is_available():
#             model.cuda()
        
#         # Step 5: Set vocabulary and configure model
#         print("⚙️ Configuring model...")
#         model.set_vocab(vocab, names=names)
#         model.model.model[-1].is_fused = True
#         model.model.model[-1].conf = 0.001  # Very low confidence threshold
#         model.model.model[-1].max_det = 1000  # High max detections
        
#         print("✅ Official prompt-free model setup complete!")
#         return model, names, unfused_model
        
#     except Exception as e:
#         print(f"❌ Error in official setup: {e}")
#         print("💡 Trying simplified approach...")
        
#         # Fallback approach
#         try:
#             model = YOLOE.from_pretrained("jameslahm/yoloe-v8l-seg")
#             if device == "cuda" and torch.cuda.is_available():
#                 model.cuda()
            
#             with open(ram_tag_file, 'r', encoding='utf-8') as f:
#                 names = [x.strip() for x in f.readlines() if x.strip()]
            
#             return model, names, None
            
#         except Exception as e2:
#             print(f"❌ Fallback also failed: {e2}")
#             return None, None, None

# def predict_official_prompt_free(model, image_path, names):
#     """
#     Perform official prompt-free detection
#     """
#     print(f"🔍 Performing official prompt-free detection on: {image_path}")
    
#     # Load image
#     image = Image.open(image_path)
    
#     # Perform detection (no prompts needed in prompt-free mode)
#     results = model.predict(
#         source=image,
#         verbose=True,
#         save=False
#     )
    
#     # Extract detected classes
#     detected_classes = []
#     if len(results) > 0 and results[0].boxes is not None:
#         cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
#         for cls_idx in cls_indices:
#             if cls_idx < len(names):
#                 detected_classes.append(names[cls_idx])
    
#     detected_classes = list(set(detected_classes))
    
#     return results, image, detected_classes

# def visualize_official_results(image, results, names, detected_classes, save_path=None):
#     """
#     Visualize official prompt-free detection results
#     """
#     img_array = np.array(image)
    
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
#     ax.imshow(img_array)
    
#     if len(results) > 0 and results[0].boxes is not None:
#         boxes = results[0].boxes
        
#         xyxy = boxes.xyxy.cpu().numpy()
#         conf = boxes.conf.cpu().numpy()
#         cls = boxes.cls.cpu().numpy().astype(int)
        
#         print(f"🔍 Detected {len(xyxy)} objects using official method:")
#         print(f"🎯 Detected classes: {detected_classes}")
        
#         colors = plt.cm.Set3(np.linspace(0, 1, max(20, len(set(cls)))))
        
#         for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
#             x1, y1, x2, y2 = box
#             width = x2 - x1
#             height = y2 - y1
            
#             if class_id < len(names):
#                 class_name = names[class_id]
#             else:
#                 class_name = f"Unknown_{class_id}"
            
#             color = colors[class_id % len(colors)]
            
#             # Draw bounding box
#             rect = patches.Rectangle(
#                 (x1, y1), width, height,
#                 linewidth=3, edgecolor=color, facecolor='none'
#             )
#             ax.add_patch(rect)
            
#             # Add label
#             label = f"{class_name}: {confidence:.3f}"
#             ax.text(
#                 x1, y1 - 8, label,
#                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
#                 fontsize=11, color='black', weight='bold'
#             )
            
#             print(f"  {i+1}. {class_name}: {confidence:.3f}")
#     else:
#         print("🚫 No objects detected with official method")
#         print("💡 This might indicate:")
#         print("   - Model setup issue")
#         print("   - Image doesn't contain recognizable objects")
#         print("   - Need to lower confidence threshold further")
    
#     ax.set_title("YOLOE Official Prompt-Free Detection (RAM Vocabulary)", 
#                 fontsize=16, weight='bold', pad=20)
#     ax.axis('off')
    
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
#         print(f"💾 Result saved: {save_path}")
    
#     plt.tight_layout()
#     plt.show()
#     plt.close()

# def main():
#     """
#     Main function following official approach
#     """
#     # Configuration
#     IMAGE_PATH = "sample.png"  # or try "ultralytics/assets/bus.jpg"
#     SAVE_PATH = "sample_official_detected.png"
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     print("🚀 YOLOE Official Prompt-Free Detection")
#     print("=" * 50)
#     print("📖 Following THU-MIG/yoloe official implementation")
#     print("=" * 50)
    
#     try:
#         # Setup official model
#         model, names, unfused_model = setup_official_prompt_free_model(IMAGE_PATH, DEVICE)
        
#         if model is None:
#             print("❌ Failed to setup model. Exiting.")
#             return
        
#         # Perform detection
#         results, image, detected_classes = predict_official_prompt_free(model, IMAGE_PATH, names)
        
#         # Visualize results
#         print("🎨 Visualizing results...")
#         visualize_official_results(image, results, names, detected_classes, SAVE_PATH)
        
#         print("✅ Official prompt-free detection completed!")
#         if detected_classes:
#             print(f"🎉 Successfully detected: {', '.join(detected_classes)}")
#         else:
#             print("🤔 No detections. Debug tips:")
#             print("   - Check if model files are properly downloaded")
#             print("   - Try with 'ultralytics/assets/bus.jpg'")
#             print("   - Verify RAM tag list is loaded correctly")
#             print("   - Model might need specific vocabulary setup")
        
#     except FileNotFoundError:
#         print(f"❌ Image not found: {IMAGE_PATH}")
#         print("💡 Try with 'ultralytics/assets/bus.jpg' or ensure image exists")
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg-pf.pt")

# Run prediction. No prompts required.
results = model.predict("sample.png")

# Show results
results[0].show()