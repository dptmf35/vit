#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
import torch

class InteractiveFurnitureDetector(Node):
    def __init__(self):
        super().__init__('interactive_furniture_detector')
        
        # ROS2 설정
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/stereo_image_color',
            self.image_callback,
            10
        )
        
        # 상태 변수
        self.first_frame = True
        self.visual_prompts_ready = False
        self.current_image = None
        self.visual_prompts = None
        
        # 박스 그리기 상태
        self.drawing = False
        self.boxes = []
        self.current_box = None
        self.box_labels = []
        
        # 침실 가구 클래스 정의
        self.furniture_classes = [
            "bed", "lamp", "window", "curtain", 
            "nightstand", "closet", "hanger"
        ]
        
        # YOLOE 모델 초기화
        self.get_logger().info("YOLOE 모델 로딩 중...")
        try:
            self.model = YOLOE("yoloe-v8l-seg.pt")
            self.get_logger().info("YOLOE 모델 로딩 완료")
        except Exception as e:
            self.get_logger().error(f"모델 로딩 실패: {e}")
            return
        
        # 시각화 설정
        self.colors = self.generate_colors(len(self.furniture_classes))
        
        self.get_logger().info("대화형 가구 탐지기 초기화 완료")
        self.get_logger().info("첫 번째 프레임에서 박스를 그려 객체를 지정하세요")

    def generate_colors(self, num_classes):
        """클래스별 고유 색상 생성"""
        colors = []
        for i in range(num_classes):
            hue = i * 180 / num_classes
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors

    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백 함수 - 박스 그리기"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y, x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_box is not None:
                self.current_box[2] = x
                self.current_box[3] = y
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_box is not None:
                self.drawing = False
                self.current_box[2] = x
                self.current_box[3] = y
                
                # 박스가 유효한 크기인지 확인
                if abs(self.current_box[2] - self.current_box[0]) > 10 and \
                   abs(self.current_box[3] - self.current_box[1]) > 10:
                    
                    # 클래스 선택 다이얼로그
                    selected_class = self.select_class()
                    if selected_class is not None:
                        self.boxes.append(self.current_box.copy())
                        self.box_labels.append(selected_class)
                        self.get_logger().info(f"박스 {len(self.boxes)}: {self.furniture_classes[selected_class]} 추가됨")
                
                self.current_box = None

    def select_class(self):
        """클래스 선택 함수"""
        print("\n=== 객체 클래스 선택 ===")
        for i, class_name in enumerate(self.furniture_classes):
            print(f"{i}: {class_name}")
        
        try:
            choice = int(input("클래스 번호를 입력하세요 (0-{}): ".format(len(self.furniture_classes)-1)))
            if 0 <= choice < len(self.furniture_classes):
                return choice
            else:
                print("잘못된 번호입니다.")
                return None
        except ValueError:
            print("숫자를 입력해주세요.")
            return None

    def image_callback(self, msg):
        """ROS2 이미지 콜백 함수"""
        try:
            # ROS Image -> OpenCV 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image.copy()
            
            if self.first_frame:
                self.handle_first_frame(cv_image)
            else:
                if self.visual_prompts_ready:
                    self.detect_with_visual_prompts(cv_image)
                    
        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {e}")

    def handle_first_frame(self, image):
        """첫 번째 프레임 처리 - 사용자 박스 입력"""
        display_image = image.copy()
        
        # 현재 그리고 있는 박스 그리기
        if self.drawing and self.current_box is not None:
            cv2.rectangle(display_image, 
                         (self.current_box[0], self.current_box[1]), 
                         (self.current_box[2], self.current_box[3]), 
                         (0, 255, 0), 2)
        
        # 완성된 박스들 그리기
        for i, (box, label) in enumerate(zip(self.boxes, self.box_labels)):
            color = self.colors[label]
            cv2.rectangle(display_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # 라벨 표시
            class_name = self.furniture_classes[label]
            cv2.putText(display_image, f"{i+1}: {class_name}", 
                       (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 안내 메시지
        cv2.putText(display_image, "마우스로 객체 박스를 그리세요", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_image, "'q': 종료, 'c': 박스 완료, 'r': 리셋", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_image, f"박스 개수: {len(self.boxes)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Visual Prompt Setup - First Frame', display_image)
        cv2.setMouseCallback('Visual Prompt Setup - First Frame', self.mouse_callback)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("프로그램 종료")
            rclpy.shutdown()
        elif key == ord('c') and len(self.boxes) > 0:
            self.complete_visual_prompt_setup()
        elif key == ord('r'):
            self.reset_boxes()

    def reset_boxes(self):
        """박스 리셋"""
        self.boxes = []
        self.box_labels = []
        self.get_logger().info("모든 박스가 리셋되었습니다")

    def complete_visual_prompt_setup(self):
        """Visual prompt 설정 완료"""
        if len(self.boxes) == 0:
            self.get_logger().warning("박스가 없습니다. 최소 1개 이상 그려주세요.")
            return
        
        try:
            # Visual prompts 생성
            bboxes_array = np.array(self.boxes, dtype=np.float32)
            cls_array = np.array(self.box_labels, dtype=np.int32)
            
            self.visual_prompts = dict(
                bboxes=[bboxes_array],
                cls=[cls_array]
            )
            
            self.visual_prompts_ready = True
            self.first_frame = False
            
            cv2.destroyWindow('Visual Prompt Setup - First Frame')
            
            self.get_logger().info(f"Visual prompt 설정 완료! {len(self.boxes)}개 객체")
            self.get_logger().info("이제 자동 탐지를 시작합니다...")
            
            # 설정 요약 출력
            for i, (box, label) in enumerate(zip(self.boxes, self.box_labels)):
                class_name = self.furniture_classes[label]
                self.get_logger().info(f"객체 {i+1}: {class_name} at [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
                
        except Exception as e:
            self.get_logger().error(f"Visual prompt 설정 실패: {e}")

    def detect_with_visual_prompts(self, image):
        """Visual prompt를 사용한 탐지"""
        try:
            # 이미지 크기 확인
            img_height, img_width = image.shape[:2]
            self.get_logger().debug(f"이미지 크기: {img_width}x{img_height}")
            
            # YOLOE visual prompt 예측
            results = self.model.predict(
                source=image,
                prompts=self.visual_prompts,
                predictor=YOLOEVPSegPredictor,
                conf=0.25,  # 낮은 신뢰도로 시작
                iou=0.5,
                save=False,
                verbose=False,
                imgsz=(640, 640)  # 고정 입력 크기 지정
            )
            
            if results and len(results) > 0:
                # 결과 시각화
                annotated_image = self.visualize_detection_results(image, results[0])
                
                # 결과 로깅
                self.log_detection_results(results[0])
                
                # 이미지 표시
                cv2.imshow('Furniture Detection - Live', annotated_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("탐지 종료")
                    rclpy.shutdown()
                elif key == ord('r'):
                    self.restart_setup()
            else:
                # 탐지 결과가 없을 때도 이미지 표시
                display_image = image.copy()
                cv2.putText(display_image, "탐지된 객체 없음", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_image, "'q': 종료, 'r': 재설정", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Furniture Detection - Live', display_image)
                cv2.waitKey(1)
                    
        except Exception as e:
            self.get_logger().error(f"탐지 오류: {e}")
            # 에러 발생시에도 원본 이미지는 표시
            try:
                display_image = image.copy()
                cv2.putText(display_image, f"오류: {str(e)[:50]}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(display_image, "'q': 종료, 'r': 재설정", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Furniture Detection - Live', display_image)
                cv2.waitKey(1)
            except:
                pass

    def restart_setup(self):
        """설정 다시 시작"""
        self.first_frame = True
        self.visual_prompts_ready = False
        self.boxes = []
        self.box_labels = []
        self.visual_prompts = None
        cv2.destroyAllWindows()
        self.get_logger().info("설정을 다시 시작합니다...")

    def visualize_detection_results(self, image, result):
        """탐지 결과 시각화"""
        annotated_image = image.copy()
        img_height, img_width = image.shape[:2]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                if cls_id < len(self.box_labels):
                    original_cls_id = self.box_labels[cls_id]
                    if original_cls_id < len(self.furniture_classes):
                        x1, y1, x2, y2 = box.astype(int)
                        color = self.colors[original_cls_id]
                        class_name = self.furniture_classes[original_cls_id]
                        
                        # 바운딩 박스
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # 라벨
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 세그멘테이션 마스크 - 크기 문제 해결
        if result.masks is not None:
            try:
                masks = result.masks.data.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, mask in enumerate(masks):
                    if i < len(class_ids):
                        cls_id = class_ids[i]
                        if cls_id < len(self.box_labels):
                            original_cls_id = self.box_labels[cls_id]
                            if original_cls_id < len(self.furniture_classes):
                                color = self.colors[original_cls_id]
                                
                                # 마스크 크기 확인 및 리사이즈
                                mask_height, mask_width = mask.shape
                                if mask_height != img_height or mask_width != img_width:
                                    # 마스크를 이미지 크기에 맞게 리사이즈
                                    mask = cv2.resize(mask.astype(np.uint8), (img_width, img_height))
                                    mask = mask.astype(bool)
                                
                                # 컬러 마스크 생성
                                colored_mask = np.zeros_like(annotated_image)
                                colored_mask[mask > 0.5] = color
                                annotated_image = cv2.addWeighted(annotated_image, 0.7, colored_mask, 0.3, 0)
                                
            except Exception as mask_error:
                # 마스크 처리 중 에러가 발생해도 바운딩 박스는 표시
                self.get_logger().warning(f"마스크 처리 중 오류 (무시됨): {mask_error}")
        
        # 안내 메시지
        cv2.putText(annotated_image, "'q': 종료, 'r': 재설정", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_image

    def log_detection_results(self, result):
        """탐지 결과 로깅"""
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            detected_objects = {}
            for conf, cls_id in zip(confidences, class_ids):
                if cls_id < len(self.box_labels):
                    original_cls_id = self.box_labels[cls_id]
                    if original_cls_id < len(self.furniture_classes):
                        class_name = self.furniture_classes[original_cls_id]
                        if class_name not in detected_objects:
                            detected_objects[class_name] = []
                        detected_objects[class_name].append(conf)
            
            if detected_objects:
                summary = []
                for obj_class, confs in detected_objects.items():
                    avg_conf = np.mean(confs)
                    count = len(confs)
                    summary.append(f"{obj_class}: {count}개 (평균: {avg_conf:.2f})")
                
                self.get_logger().info(f"탐지됨: {', '.join(summary)}")

def main(args=None):
    rclpy.init(args=args)
    
    print("\n=== 대화형 가구 탐지기 ===")
    print("1. 첫 번째 프레임에서 마우스로 객체 박스를 그리세요")
    print("2. 각 박스마다 클래스를 선택하세요")
    print("3. 'c' 키로 설정을 완료하세요")
    print("4. 이후 프레임에서 자동 탐지가 시작됩니다")
    print("===============================\n")
    
    try:
        detector = InteractiveFurnitureDetector()
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()