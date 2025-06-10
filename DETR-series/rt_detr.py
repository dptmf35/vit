import torch
from transformers import RTDetrImageProcessor, RTDetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import numpy as np
import matplotlib.pyplot as plt

class RTDETRPredictor:
    """RT-DETR 모델을 사용한 객체 탐지 예측기"""
    
    def __init__(self, model_name="PekingU/rtdetr_r50vd", device=None):
        """
        Args:
            model_name (str): 사용할 RT-DETR 모델명
                - "PekingU/rtdetr_r18vd" (가장 빠름)
                - "PekingU/rtdetr_r50vd" (권장)
                - "PekingU/rtdetr_r101vd" (가장 정확함)
            device: 사용할 디바이스 (None이면 자동 선택)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델과 프로세서 로드
        print(f"Loading {model_name} on {self.device}...")
        self.processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.model = RTDetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO 클래스 이름
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def load_image(self, image_path):
        """이미지 로드 (로컬 파일 또는 URL)"""
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        return image.convert('RGB')
    
    def predict(self, image, confidence_threshold=0.5):
        """
        이미지에서 객체 탐지 수행
        
        Args:
            image: PIL Image 또는 이미지 경로
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            dict: 예측 결과 (boxes, scores, labels)
        """
        if isinstance(image, str):
            image = self.load_image(image)
        
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 예측 수행
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 후처리
        target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        # 결과 정리
        predictions = {
            'boxes': results['boxes'].cpu().numpy(),
            'scores': results['scores'].cpu().numpy(),
            'labels': results['labels'].cpu().numpy(),
            'class_names': [self.coco_classes[label] for label in results['labels']]
        }
        
        return predictions, image
    
    def visualize_predictions(self, image, predictions, save_path=None, show=True):
        """예측 결과 시각화"""
        draw = ImageDraw.Draw(image)
        
        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions['boxes'])))
        
        for i, (box, score, label_name) in enumerate(zip(
            predictions['boxes'], 
            predictions['scores'], 
            predictions['class_names']
        )):
            x1, y1, x2, y2 = box
            color = tuple(int(c * 255) for c in colors[i][:3])
            
            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 라벨과 신뢰도 텍스트
            text = f"{label_name}: {score:.2f}"
            
            # 텍스트 배경 박스
            bbox = draw.textbbox((x1, y1-25), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1-25), text, fill='white', font=font)
        
        if save_path:
            image.save(save_path)
            print(f"결과 이미지 저장: {save_path}")
        
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.title('RT-DETR Object Detection Results')
            plt.show()
        
        return image
    
    def predict_and_visualize(self, image_path, confidence_threshold=0.5, 
                            save_path=None, show=True):
        """예측과 시각화를 한번에 수행"""
        predictions, image = self.predict(image_path, confidence_threshold)
        
        print(f"탐지된 객체 수: {len(predictions['boxes'])}")
        for i, (class_name, score) in enumerate(zip(predictions['class_names'], predictions['scores'])):
            print(f"  {i+1}. {class_name}: {score:.3f}")
        
        self.visualize_predictions(image, predictions, save_path, show)
        
        return predictions, image


# RT-DETRv2 사용을 위한 클래스 (동일한 인터페이스)
class RTDETRv2Predictor(RTDETRPredictor):
    """RT-DETRv2 모델을 사용한 객체 탐지 예측기"""
    
    def __init__(self, model_name="jadechoghari/RT-DETRv2", device=None):
        super().__init__(model_name, device)


# 사용 예시
if __name__ == "__main__":
    # RT-DETR 모델 초기화
    predictor = RTDETRPredictor("PekingU/rtdetr_r50vd")
    
    # 예시 이미지 URL (또는 로컬 파일 경로 사용 가능)
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    
    # 예측 및 시각화
    predictions, result_image = predictor.predict_and_visualize(
        image_url,
        confidence_threshold=0.5,
        save_path="rt_detr_result.jpg"
    )
    
    # RT-DETRv2 사용 예시
    print("\n" + "="*50)
    print("RT-DETRv2로 동일한 이미지 예측:")
    
    predictor_v2 = RTDETRv2Predictor("jadechoghari/RT-DETRv2")
    predictions_v2, result_image_v2 = predictor_v2.predict_and_visualize(
        image_url,
        confidence_threshold=0.5,
        save_path="rt_detr_v2_result.jpg"
    )
    