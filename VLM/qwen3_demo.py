from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import cv2, ast 
import torch 

# default: Load the model on the available device(s)
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./test.png",
            },
            {"type": "text", "text": "Locate the center of the bed in this image. Return the normalized coordinates in the format of (x, y) between 0 and 1."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


point_norm = ast.literal_eval(output_text[0])


image = cv2.imread("./test.png")
h, w, _ = image.shape

center_x = int(point_norm[0] * w)
center_y = int(point_norm[1] * h)

cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()