from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11s-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["table","fridge","chair","dish","gas stove","microwave"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict(
                        "sample.png",
                        conf=0.1,
                        iou=0.3,
                        max_det=100,
                        agnostic_nms=False
                        )

# Show results
results[0].show()