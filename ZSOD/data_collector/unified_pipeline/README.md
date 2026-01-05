# Unified ML Pipeline - Data Collection to Deployment

완전 통합된 머신러닝 파이프라인으로 데이터 수집부터 ROS2 배포까지 GUI를 통해 관리할 수 있습니다.

## 주요 기능

### 📸 데이터 수집 (Data Collection)
- ROS2 카메라 토픽에서 자동으로 이미지 수집
- YOLOE 또는 학습된 YOLO11 모델 사용
- 실시간 detection 및 어노테이션 자동 생성
- 수집 파라미터 커스터마이징 (confidence threshold, interval 등)
- 테스트 모드 지원

### ✏️ 어노테이션 검증 및 수정 (Annotation Review)
- GUI 기반 어노테이션 검증
- 시각적 편집 도구
- 라벨 형식 검증
- 데이터셋 통계 분석
- 불필요한 데이터 삭제

### 🎓 모델 학습 (Training)
- YOLO11 모델 학습 (n/s/m/l/x 크기 선택 가능)
- 자동 데이터셋 분할 (train/val)
- 소규모 데이터셋 최적화
- 실시간 학습 진행 상황 모니터링
- 학습 결과 분석

### 📊 모델 평가 (Evaluation)
- 학습된 모델 성능 평가
- 단일 이미지 테스트
- 디렉토리 배치 테스트
- mAP, Precision, Recall 등 메트릭 제공
- 클래스별 성능 분석

### 🚀 ROS2 배포 (Deployment)
- 학습된 모델을 ROS2 노드로 배포
- 실시간 객체 감지
- 어노테이션된 이미지를 새로운 토픽으로 퍼블리시
- Detection 결과 JSON 형식으로 퍼블리시
- Bounding box 정보 퍼블리시

## 설치 및 요구사항

### 필수 패키지

```bash
# ROS2 (Humble 권장)
source /opt/ros/humble/setup.bash

# Python 패키지
pip install ultralytics opencv-python numpy pyyaml pandas

# GUI 패키지
sudo apt-get install python3-tk

# ROS2 Python 패키지
pip install cv-bridge
```

## 디렉토리 구조

```
unified_pipeline/
├── main.py                      # 메인 진입점
├── README.md                    # 이 파일
├── config/
│   ├── pipeline_config.py       # 설정 관리
│   └── pipeline_config.json     # 저장된 설정 (자동 생성)
├── modules/
│   ├── data_collection.py       # 데이터 수집 모듈
│   ├── annotation_review.py     # 어노테이션 검증 모듈
│   ├── training.py              # 학습 모듈
│   ├── evaluation.py            # 평가 모듈
│   └── deployment.py            # 배포 모듈
├── gui/
│   └── main_window.py           # GUI 메인 윈도우
└── logs/                        # 로그 파일 (자동 생성)
```

## 사용 방법

### 1. GUI 실행

```bash
cd /path/to/ZSOD/data_collector/unified_pipeline
python3 main.py
```

### 2. 설정 (Configuration)

**⚙️ Configuration 탭**에서:

1. **Target Classes 설정**
   - 수집하고 학습할 객체 클래스를 쉼표로 구분하여 입력
   - 예: `chair, table, bed, sofa, tv`

2. **Collection Settings 설정**
   - Confidence Threshold: 객체 인식 신뢰도 임계값 (0.1 ~ 1.0)
   - Collection Interval: 데이터 수집 간격 (초)
   - Dataset Path: 데이터셋 저장 경로
   - Use YOLO11: 학습된 YOLO11 모델 사용 여부

3. **Training Settings 설정**
   - Model Size: YOLO11 모델 크기 선택
   - Epochs: 학습 에폭 수

4. **설정 저장**
   - "💾 Save Configuration" 버튼 클릭

### 3. 데이터 수집 (Data Collection)

**📸 Data Collection 탭**에서:

1. **테스트 모드로 시작** (권장)
   - "🔍 Test Mode" 버튼 클릭
   - ROS2 토픽에서 이미지를 받아 detection만 수행
   - 데이터 저장 안함

2. **수집 모드로 시작**
   - "▶️ Start Collection" 버튼 클릭
   - 실시간으로 이미지와 어노테이션 수집 시작
   - 수집 통계 확인

3. **수집 중지**
   - "⏹️ Stop Collection" 버튼 클릭

4. **데이터셋 정보 확인**
   - "📊 Dataset Info" 버튼으로 수집된 데이터 확인

### 4. 어노테이션 검증 (Annotation Review)

**✏️ Annotation Review 탭**에서:

1. **리뷰어 실행**
   - "🔍 Launch Reviewer" 버튼 클릭
   - 별도 창에서 어노테이션 검증 GUI 실행

2. **데이터셋 통계**
   - "📊 Dataset Statistics" 버튼으로 전체 통계 확인

3. **라벨 검증**
   - "✅ Validate Labels" 버튼으로 형식 오류 확인

### 5. 모델 학습 (Training)

**🎓 Training 탭**에서:

1. **데이터셋 분석**
   - "📊 Dataset Analysis" 버튼으로 데이터셋 품질 확인
   - 권장사항 확인

2. **학습 시작**
   - "▶️ Start Training" 버튼 클릭
   - 진행 상황 모니터링
   - 로그 확인

3. **학습 결과 확인**
   - "📈 Training Results" 버튼으로 메트릭 확인

### 6. 모델 평가 (Evaluation)

**📊 Evaluation 탭**에서:

1. **데이터셋 평가**
   - "📊 Evaluate on Dataset" 버튼으로 validation set 평가

2. **단일 이미지 테스트**
   - "🖼️ Test on Image" 버튼으로 이미지 선택 및 테스트

3. **디렉토리 테스트**
   - "📁 Test on Directory" 버튼으로 여러 이미지 배치 테스트

### 7. ROS2 배포 (Deployment)

**🚀 Deployment 탭**에서:

1. **모델 경로 설정**
   - 학습된 모델 경로 확인 또는 변경
   - Browse 버튼으로 모델 선택

2. **카메라 토픽 설정**
   - ROS2 카메라 토픽 입력

3. **배포 시작**
   - "▶️ Start Deployment" 버튼 클릭
   - ROS2 노드가 백그라운드에서 실행됨

4. **결과 확인**
   - ROS2 토픽으로 결과 확인:
     - `/custom_yolo/annotated_image` - 어노테이션된 이미지
     - `/custom_yolo/detections` - Detection 결과 (JSON)
     - `/custom_yolo/bounding_boxes` - Bounding box 정보

5. **배포 중지**
   - "⏹️ Stop Deployment" 버튼 클릭

## 전체 워크플로우 예시

### 완전 자동화 파이프라인

```
1. Configuration 설정
   ↓
2. Data Collection (수집)
   - 카메라 토픽에서 자동 수집
   - 100~500개 이미지 수집
   ↓
3. Annotation Review (검증)
   - 수집된 어노테이션 검증
   - 오류 수정 및 불필요한 데이터 삭제
   ↓
4. Training (학습)
   - 데이터셋 분석
   - YOLO11 모델 학습
   - 100~200 epochs
   ↓
5. Evaluation (평가)
   - 학습된 모델 성능 평가
   - 테스트 이미지로 검증
   ↓
6. Deployment (배포)
   - ROS2 노드로 배포
   - 실시간 객체 감지 서비스
```

## ROS2 토픽 정보

### 입력 토픽
- `/stereo_image_color` (또는 설정한 카메라 토픽)
  - 타입: `sensor_msgs/Image`
  - 원본 카메라 이미지

### 출력 토픽 (Collection)
- `/yolo_detection_rviz` (YOLOE)
- `/yolo11_detection_rviz` (YOLO11)
  - 타입: `sensor_msgs/Image`
  - Detection 결과 시각화

### 출력 토픽 (Deployment)
- `/custom_yolo/annotated_image`
  - 타입: `sensor_msgs/Image`
  - 어노테이션된 이미지

- `/custom_yolo/detections`
  - 타입: `std_msgs/String`
  - JSON 형식 detection 결과
  ```json
  [
    {
      "class_id": 0,
      "class_name": "chair",
      "confidence": 0.85,
      "bbox": [100, 200, 300, 400]
    }
  ]
  ```

- `/custom_yolo/bounding_boxes`
  - 타입: `std_msgs/Float32MultiArray`
  - Bounding box 배열: [class_id, conf, x1, y1, x2, y2, ...]

## 설정 파일

설정은 `config/pipeline_config.json`에 자동으로 저장됩니다:

```json
{
  "collection": {
    "target_classes": ["chair", "table", "bed"],
    "conf_threshold": 0.6,
    "collection_interval": 2.0,
    "dataset_path": "~/yolo_dataset"
  },
  "training": {
    "model_size": "yolo11s.pt",
    "epochs": 100
  },
  "evaluation": {
    "conf_threshold": 0.25
  },
  "deployment": {
    "model_path": "train_model/training_output/train/weights/best.pt",
    "camera_topic": "/stereo_image_color"
  }
}
```

## 문제 해결

### 일반적인 문제

1. **GUI가 실행되지 않음**
   ```bash
   sudo apt-get install python3-tk
   ```

2. **ROS2 토픽을 받지 못함**
   ```bash
   # 토픽 확인
   ros2 topic list

   # 토픽 정보 확인
   ros2 topic info /stereo_image_color
   ```

3. **모델 로딩 실패**
   - 모델 파일 경로 확인
   - 절대 경로 사용 권장

4. **학습이 너무 느림**
   - GPU 사용 확인: `nvidia-smi`
   - 배치 크기 조정
   - 더 작은 모델 사용 (yolo11n.pt)

5. **메모리 부족**
   - 배치 크기 줄이기
   - 이미지 크기 줄이기 (640 → 416)

## 고급 사용법

### 커스텀 클래스 사용

기본 제공되는 클래스 외에 원하는 객체 클래스를 추가할 수 있습니다:

1. Configuration 탭에서 원하는 클래스 입력
2. 해당 클래스가 포함된 이미지 수집
3. 학습 진행

### 배치 모드 사용

GUI 없이 명령줄에서 각 모듈을 직접 사용할 수 있습니다:

```python
from modules.data_collection import DataCollectionModule
from config.pipeline_config import CollectionConfig

config = CollectionConfig(
    target_classes=["chair", "table"],
    conf_threshold=0.6
)

module = DataCollectionModule(config)
module.start_collection(test_mode=False)
```

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 연락처

질문이나 문의사항이 있으시면 이슈를 통해 연락해주세요.
