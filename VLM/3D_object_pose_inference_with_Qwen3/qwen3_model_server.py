#!/usr/bin/env python3
"""
Qwen3 Model Server - Runs in conda environment
Receives image and object name, returns normalized coordinates
"""

import torch
import ast
import base64
import io
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image as PILImage
import numpy as np
import zmq


class Qwen3ModelServer:
    def __init__(self, port=5555):
        self.port = port
        
        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        # Initialize Qwen3 model
        print("Loading Qwen3 model...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        print("Model loaded successfully!")
        print(f"Server listening on port {port}...")
    
    def locate_object(self, image_array, object_name):
        """
        Locate object center using Qwen3 model
        
        Args:
            image_array: numpy array (BGR format from cv2)
            object_name: name of the object to locate
            
        Returns:
            tuple: (x, y) normalized coordinates or None if failed
        """
        try:
            # Convert BGR to RGB
            image_rgb = image_array[:, :, ::-1]
            pil_image = PILImage.fromarray(image_rgb)
            
            # Prepare messages for model
            
            messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "text",
                    "text": f"""Find the center point of the {object_name} in this image.

IMPORTANT: Return ONLY normalized coordinates as a tuple (x, y) where both values are between 0.0 and 1.0.
- x=0.0 is left edge, x=1.0 is right edge
- y=0.0 is top edge, y=1.0 is bottom edge
- The center of the image is (0.5, 0.5)

Examples of correct format:
- Object at center: (0.5, 0.5)
- Object at top-left: (0.2, 0.3)
- Object at bottom-right: (0.8, 0.75)

Do NOT use pixel coordinates. Use only decimal values between 0 and 1.

Answer with ONLY the tuple, nothing else. Format: (x, y)"""
                    },
                ],
            }
        ]
            
            # Preparation for inference
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Inference
            print(f"Locating {object_name}...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            print(f"Model output: {output_text[0]}")
            
            # Parse normalized coordinates
            point_norm = ast.literal_eval(output_text[0])
            return point_norm
            
        except Exception as e:
            print(f"Error in locate_object: {str(e)}")
            return None
    
    def run(self):
        """Main server loop"""
        print("\n========================================")
        print("Qwen3 Model Server Running")
        print("Waiting for requests...")
        print("========================================\n")
        
        while True:
            try:
                # Wait for request
                message = self.socket.recv_json()
                
                # Parse request
                image_data = message['image']
                object_name = message['object_name']
                
                # Decode image
                image_bytes = base64.b64decode(image_data)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image_array = image_array.reshape(message['shape'])
                
                # Process request
                result = self.locate_object(image_array, object_name)
                
                # Send response
                if result is not None:
                    response = {
                        'success': True,
                        'coordinates': result
                    }
                else:
                    response = {
                        'success': False,
                        'error': 'Failed to locate object'
                    }
                
                self.socket.send_json(response)
                
            except KeyboardInterrupt:
                print("\nShutting down server...")
                break
            except Exception as e:
                print(f"Error processing request: {str(e)}")
                response = {
                    'success': False,
                    'error': str(e)
                }
                self.socket.send_json(response)
        
        self.socket.close()
        self.context.term()


def main():
    server = Qwen3ModelServer(port=5555)
    server.run()


if __name__ == '__main__':
    main()

