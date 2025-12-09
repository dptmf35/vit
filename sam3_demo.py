import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
# model = build_sam3_image_model()
# processor = Sam3Processor(model)
# Load an image
# image = Image.open("<YOUR_IMAGE_PATH.jpg>")
# inference_state = processor.set_image(image)
# # Prompt the model with text
# output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# # Get the masks, bounding boxes, and scores
# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### For Video ####################################

from sam3.model_builder import build_sam3_video_predictor

import cv2
import numpy as np

video_predictor = build_sam3_video_predictor()
video_path = "./assets/videos/bedroom.mp4" # a JPEG folder or an MP4 video file

# Define prompts and colors
prompts = ["person", "bed", "pillow"]
colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255]] # Green, Blue, Red

sessions = []
streams = []

print("Initializing sessions for prompts:", prompts)

# Initialize a session for each prompt
for prompt in prompts:
    # Start session
    resp = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    sess_id = resp["session_id"]
    sessions.append(sess_id)
    
    # Add prompt
    video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=sess_id,
            frame_index=0,
            text=prompt,
        )
    )
    
    # Create stream
    stream = video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=sess_id,
        )
    )
    streams.append(stream)

# Open video for reading frames
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Starting multi-session propagation...")

# Iterate through all streams simultaneously
# zip(*streams) pulls one frame output from each session
for outputs in zip(*streams):
    # All outputs should correspond to the same frame index
    frame_idx = outputs[0]["frame_index"]
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
        
    # Overlay masks from all sessions
    for i, output_dict in enumerate(outputs):
        out_obj_ids = output_dict["outputs"]["out_obj_ids"]
        out_mask = output_dict["outputs"]["out_binary_masks"]
        color = colors[i % len(colors)]
        
        if len(out_obj_ids) > 0:
            for j, obj_id in enumerate(out_obj_ids):
                mask = out_mask[j]
                colored_mask = np.zeros_like(frame)
                colored_mask[mask] = color
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

    # Show the frame
    cv2.imshow("SAM3 Multi-Object Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close all sessions
for sess_id in sessions:
    video_predictor.handle_request(request=dict(type="close_session", session_id=sess_id))

print("Done!")