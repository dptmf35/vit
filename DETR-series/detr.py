"""
Transformers 라이브러리를 사용한 DETR 이미지 객체 검출
출처: Hugging Face Transformers (https://huggingface.co/docs/transformers/model_doc/detr)
"""

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import requests
import matplotlib.pyplot as plt
import numpy as np

# COCO 클래스 이름들
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class DETRImageDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", confidence_threshold=0.7):
        """
        DETR 이미지 검출기 초기화
        
        Args:
            model_name (str): 사용할 모델명
            confidence_threshold (float): 검출 신뢰도 임계값
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"DETR 모델 로딩 중: {model_name}")
        print(f"사용 디바이스: {self.device}")
        
        # 모델과 프로세서 로드
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print("모델 로딩 완료!")
    
    def load_image(self, image_path):
        """
        이미지 로드 (로컬 파일 또는 URL)
        
        Args:
            image_path (str): 이미지 경로 또는 URL
            
        Returns:
            PIL.Image: 로드된 이미지
        """
        try:
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            
            # RGB로 변환 (RGBA나 다른 모드인 경우)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return None
    
    def detect_objects(self, image):
        """
        이미지에서 객체 검출
        
        Args:
            image (PIL.Image): 입력 이미지
            
        Returns:
            dict: 검출 결과 (boxes, labels, scores)
        """
        # 이미지 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 결과 후처리
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]
        
        return results
    
    def visualize_predictions(self, image, results, save_path=None, show_plot=True):
        """
        검출 결과를 이미지에 시각화
        
        Args:
            image (PIL.Image): 원본 이미지
            results (dict): 검출 결과
            save_path (str, optional): 저장할 경로
            show_plot (bool): matplotlib으로 표시할지 여부
            
        Returns:
            PIL.Image: 시각화된 이미지
        """
        # 이미지 복사
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # 폰트 설정 (시스템에 따라 조정)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # 검출된 객체들 그리기
        boxes = results["boxes"]
        labels = results["labels"]
        scores = results["scores"]
        
        print(f"검출된 객체 수: {len(boxes)}")
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box.tolist()
            
            # 클래스 이름과 신뢰도
            class_name = COCO_CLASSES[label.item()]
            confidence = score.item()
            
            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # 라벨 텍스트
            label_text = f"{class_name}: {confidence:.2f}"
            
            # 텍스트 배경
            text_bbox = draw.textbbox((x1, y1), label_text, font=font)
            draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], 
                         fill="red")
            
            # 텍스트
            draw.text((x1, y1), label_text, fill="white", font=font)
            
            print(f"{i+1}. {class_name}: {confidence:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        # 저장
        if save_path:
            draw_image.save(save_path)
            print(f"결과 이미지 저장됨: {save_path}")
        
        # 표시
        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.imshow(draw_image)
            plt.axis('off')
            plt.title(f'DETR Object Detection - {len(boxes)} objects detected')
            plt.tight_layout()
            plt.show()
        
        return draw_image

def demo_single_image():
    """
    단일 이미지 검출 데모
    """
    # 검출기 초기화
    detector = DETRImageDetector(confidence_threshold=0.7)
    
    # 테스트 이미지 URL (또는 로컬 파일 경로 사용 가능)
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    # 로컬 파일 사용 시: image_path = "test_image.jpg"
    
    print(f"이미지 로딩: {image_url}")
    image = detector.load_image(image_url)
    
    if image is None:
        print("이미지 로드 실패")
        return
    
    print(f"이미지 크기: {image.size}")
    
    # 객체 검출
    print("객체 검출 중...")
    results = detector.detect_objects(image)
    
    # 결과 시각화
    detector.visualize_predictions(
        image, results, 
        save_path="detection_result.jpg",
        show_plot=True
    )

def demo_batch_images():
    """
    여러 이미지 배치 처리 데모
    """
    detector = DETRImageDetector(confidence_threshold=0.5)
    
    # 여러 이미지 경로들
    image_paths = [
        "image1.jpg",
        "image2.jpg", 
        "image3.jpg"
        # 실제 존재하는 이미지 파일들로 교체하세요
    ]
    
    for i, image_path in enumerate(image_paths):
        print(f"\n=== 이미지 {i+1}/{len(image_paths)}: {image_path} ===")
        
        image = detector.load_image(image_path)
        if image is None:
            continue
            
        results = detector.detect_objects(image)
        detector.visualize_predictions(
            image, results,
            save_path=f"result_{i+1}.jpg",
            show_plot=False  # 배치 처리 시 표시 안 함
        )

def main():
    """
    메인 함수
    """
    print("=== Transformers DETR 이미지 객체 검출 ===")
    print("필요한 라이브러리: pip install transformers torch pillow matplotlib requests")
    print()
    
    try:
        # 단일 이미지 데모 실행
        demo_single_image()
        
        # 배치 처리 데모 (주석 해제하여 사용)
        # demo_batch_images()
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("다음을 확인해주세요:")
        print("1. 필요한 라이브러리가 설치되어 있는지")
        print("2. 인터넷 연결이 되어 있는지 (모델 다운로드용)")
        print("3. 이미지 파일이 존재하는지")

if __name__ == "__main__":
    main()