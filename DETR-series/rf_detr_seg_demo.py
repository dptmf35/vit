import os
import supervision as sv
# from inference import get_model
from rfdetr import RFDETRSegPreview
from rfdetr.util.coco_classes import COCO_CLASSES
from PIL import Image
from io import BytesIO
import requests

url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

# model = get_model("rfdetr-seg-preview")
model = RFDETRSegPreview()
# model.optimize_for_inference()

# predictions = model.infer(image, confidence=0.5)[0]
# detections = sv.Detections.from_inference(predictions)
detections = model.predict(image, threshold=0.5)

# labels = [prediction.class_name for prediction in predictions.predictions]
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.MaskAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)