# YOLO Dataset Collector

ROS2 기반 YOLO 데이터셋 자동 수집 도구입니다. YOLOE와 학습된 YOLO11 모델을 모두 지원합니다.

## 주요 기능

- **자동 데이터셋 수집**: 특정 객체가 인식될 때마다 자동으로 이미지와 어노테이션 저장
- **YOLO 형식 지원**: 표준 YOLO 학습 형식으로 데이터 저장 (normalized coordinates)
- **다중 모델 지원**: YOLOE (기본) 및 학습된 YOLO11 모델 지원
- **적응형 수집**: confidence threshold, IOU threshold, 수집 간격 등 세밀한 제어
- **통계 기능**: 실시간 수집 통계 및 클래스별 수집 현황 모니터링
- **인터랙티브 모드**: 런타임 중 테스트/수집 모드 전환 가능
- **시각적 편집**: 수집된 데이터의 시각적 검토 및 편집 도구

## 지원하는 객체 클래스

```
"air purifier", "bed", "cabinet", "carpet", "chair", "closet", "countertop", 
"desk", "dinningtable", "door", "fridge", "lamp", "mirror", "piano", 
"plant", "shelf", "sidetable", "sofa", "table", "tv", "tv stand", "vanity"
```

## 파일 구조

```
├── yolo_dataset_collector.py      # YOLOE 기반 메인 데이터셋 수집기
├── yolo11_dataset_collector.py    # 학습된 YOLO11 모델 기반 수집기
├── run_dataset_collector.py       # YOLOE 수집기 런처 스크립트
├── run_yolo11_collector.py        # YOLO11 수집기 런처 스크립트
├── start_collection.sh            # 통합 실행 스크립트 (메뉴 기반)
├── dataset_reviewer.py            # 데이터셋 검토 및 편집 도구
├── interactive_annotation_editor.py # 시각적 어노테이션 편집기
├── visual_prompt_detector.py      # 시각적 프롬프트 기반 감지기
├── test_visual_editor.py          # 시각적 편집기 테스트
├── toggle_mode.py                 # 모드 전환 유틸리티
└── DATASET_COLLECTOR_README.md    # 사용법 안내
```

## 설치 및 의존성

```bash
# ROS2 및 필요한 패키지들이 설치되어 있어야 합니다
pip install ultralytics opencv-python numpy

# ROS2 환경 설정
source /opt/ros/humble/setup.bash
```

## 빠른 시작

### 1. 통합 메뉴 실행 (권장)
```bash
./start_collection.sh
```

메뉴 옵션:
- **1-3**: YOLOE 기반 수집 (품질별 설정)
- **4**: YOLOE 테스트 모드
- **5**: YOLOE 인터랙티브 모드
- **6**: 커스텀 설정
- **7**: ROS2 이미지 토픽 확인
- **8**: YOLO11 학습된 모델 수집
- **9**: YOLO11 테스트 모드

### 2. 직접 실행

#### YOLOE 기반 수집
```bash
# 기본 설정으로 실행
python3 run_dataset_collector.py

# 고품질 수집
python3 run_dataset_collector.py --conf_threshold 0.8 --collection_interval 3.0

# 테스트 모드
python3 run_dataset_collector.py --test_mode
```

#### YOLO11 학습된 모델 기반 수집
```bash
# 기본 설정으로 실행 (학습된 모델 사용)
python3 run_yolo11_collector.py

# 커스텀 모델 경로
python3 run_yolo11_collector.py --model_path path/to/your/best.pt

# 테스트 모드
python3 run_yolo11_collector.py --test_mode
```

## 파라미터 설명

### YOLOE 수집기 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--conf_threshold` | 0.6 | 객체 인식 confidence threshold (0.0-1.0) |
| `--iou_threshold` | 0.4 | Non-Maximum Suppression IOU threshold |
| `--collection_interval` | 2.0 | 데이터 수집 간격 (초) |
| `--min_detections` | 1 | 저장하기 위한 최소 detection 수 |
| `--max_detections` | 50 | 이미지당 최대 detection 수 |
| `--dataset_path` | ~/yolo_dataset | 데이터셋 저장 경로 |
| `--image_topic` | /stereo_image_color | 입력 이미지 ROS 토픽 |
| `--model_path` | yoloe-11m-seg.pt | YOLOE 모델 파일 경로 |
| `--test_mode` | False | 테스트 모드 (데이터 수집 안함) |

### YOLO11 수집기 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--conf_threshold` | 0.5 | 객체 인식 confidence threshold (0.0-1.0) |
| `--iou_threshold` | 0.4 | Non-Maximum Suppression IOU threshold |
| `--collection_interval` | 2.0 | 데이터 수집 간격 (초) |
| `--min_detections` | 1 | 저장하기 위한 최소 detection 수 |
| `--max_detections` | 50 | 이미지당 최대 detection 수 |
| `--dataset_path` | ~/yolo11_dataset | 데이터셋 저장 경로 |
| `--image_topic` | /stereo_image_color | 입력 이미지 ROS 토픽 |
| `--model_path` | train_model/training_output/train/weights/best.pt | 학습된 YOLO11 모델 경로 |
| `--test_mode` | False | 테스트 모드 (데이터 수집 안함) |

## 생성되는 데이터셋 구조

### YOLOE 데이터셋
```
~/yolo_dataset/
├── images/                    # 이미지 파일들
│   ├── img_20241220_143025_123.jpg
│   ├── img_20241220_143027_456.jpg
│   └── ...
├── labels/                    # YOLO 어노테이션 파일들
│   ├── img_20241220_143025_123.txt
│   ├── img_20241220_143027_456.txt
│   └── ...
├── visualizations/            # 시각화 이미지들
│   ├── vis_20241220_143025_123.jpg
│   ├── vis_20241220_143027_456.jpg
│   └── ...
└── dataset.yaml               # YOLO 학습용 설정 파일
```

### YOLO11 데이터셋
```
~/yolo11_dataset/
├── images/                    # 이미지 파일들
├── labels/                    # YOLO 어노테이션 파일들
├── visualizations/            # 시각화 이미지들
└── dataset.yaml               # YOLO 학습용 설정 파일
```

### YOLO 어노테이션 형식
각 `.txt` 파일은 다음과 같은 형식으로 저장됩니다:
```
class_id center_x center_y width height
0 0.516667 0.544444 0.183333 0.222222
1 0.283333 0.311111 0.150000 0.166667
```
모든 좌표는 이미지 크기에 대해 정규화됩니다 (0.0-1.0).

## 모니터링 및 통계

실행 중에 다음과 같은 정보가 표시됩니다:

```
[INFO] YOLO Dataset Collector initialized - DATA COLLECTION MODE
[INFO] Model path: yoloe-11m-seg.pt
[INFO] Detection parameters: conf=0.6, iou=0.4
[INFO] Collection interval: 2.0s
[INFO] Target classes: air purifier, bed, cabinet, carpet, chair, ...
[INFO] Saved dataset sample 1: img_20241220_143025_123.jpg
[INFO] Collected data with classes: table, chair
[INFO] === Collection Statistics (Total: 10) ===
[INFO]   table: 6
[INFO]   chair: 4
[INFO]   tv: 2
```

## 인터랙티브 모드

런타임 중 키보드로 모드를 전환할 수 있습니다:

- **'t'**: 테스트 모드로 전환 (데이터 수집 안함)
- **'c'**: 수집 모드로 전환 (데이터 수집 시작)
- **'s'**: 현재 상태 및 통계 표시
- **'q'**: 프로그램 종료

## 데이터셋 검토 및 편집

### 데이터셋 리뷰어 실행
```bash
python3 run_dataset_reviewer.py ~/yolo_dataset
```

### 시각적 편집기 테스트
```bash
python3 test_visual_editor.py
```

## 최적화 팁

### 고품질 데이터셋을 위한 설정
```bash
# YOLOE
python3 run_dataset_collector.py \
    --conf_threshold 0.8 \
    --iou_threshold 0.3 \
    --collection_interval 3.0 \
    --min_detections 2

# YOLO11
python3 run_yolo11_collector.py \
    --conf_threshold 0.7 \
    --collection_interval 3.0 \
    --min_detections 2
```

### 대량 수집을 위한 설정
```bash
# YOLOE
python3 run_dataset_collector.py \
    --conf_threshold 0.5 \
    --collection_interval 1.0 \
    --max_detections 100

# YOLO11
python3 run_yolo11_collector.py \
    --conf_threshold 0.4 \
    --collection_interval 1.0 \
    --max_detections 100
```

## ROS2 토픽 및 서비스

### YOLOE 수집기
- **입력 토픽**: `/stereo_image_color` (이미지)
- **출력 토픽**: `/yolo_detection_rviz` (시각화)
- **상태 토픽**: `/collector_mode_status` (모드 상태)
- **서비스**: `/toggle_collection_mode` (모드 전환)

### YOLO11 수집기
- **입력 토픽**: `/stereo_image_color` (이미지)
- **출력 토픽**: `/yolo11_detection_rviz` (시각화)
- **상태 토픽**: `/yolo11_collector_mode_status` (모드 상태)
- **서비스**: `/toggle_yolo11_collection_mode` (모드 전환)

## 주의사항

1. **모델 파일**: 
   - YOLOE: `yoloe-11m-seg.pt` 파일이 필요합니다
   - YOLO11: `train_model/training_output/train/weights/best.pt` 파일이 필요합니다
2. **ROS2 토픽**: 지정한 이미지 토픽이 활성화되어 있어야 합니다
3. **저장 공간**: 대량 수집 시 충분한 디스크 공간을 확보하세요
4. **수집 품질**: threshold를 너무 낮게 설정하면 저품질 데이터가 수집될 수 있습니다
5. **동시 실행**: YOLOE와 YOLO11 수집기를 동시에 실행할 수 있습니다 (다른 토픽 사용)

## 수집 중단 및 재개

- `Ctrl+C`로 안전하게 중단할 수 있습니다
- 중단 시 마지막 통계가 출력됩니다
- 재시작하면 기존 데이터셋에 추가로 수집됩니다 (덮어쓰지 않음)

## 트러블슈팅

### 일반적인 문제들

1. **모델 로딩 실패**
   ```
   ERROR: Cannot load model yoloe-11m-seg.pt
   ```
   → 모델 파일 경로를 확인하세요

2. **토픽 수신 실패**
   ```
   No images received on topic /stereo_image_color
   ```
   → `ros2 topic list`로 토픽 이름을 확인하세요

3. **권한 오류**
   ```
   Permission denied: ~/yolo_dataset
   ```
   → 저장 경로의 쓰기 권한을 확인하세요

4. **Import 오류**
   ```
   Error: Cannot import yolo_dataset_collector.py
   ```
   → Python 경로 문제입니다. 스크립트가 수정되어 해결되었습니다

5. **GLIBCXX 버전 오류**
   ```
   GLIBCXX_3.4.30' not found
   ```
   → ROS2와 Anaconda 환경 충돌입니다. 시스템 Python을 사용하거나 환경 변수를 설정하세요

## YOLO 모델 학습

수집된 데이터셋은 바로 YOLO 학습에 사용할 수 있습니다:

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 학습 실행
results = model.train(data='~/yolo_dataset/dataset.yaml', epochs=100)
```

## 업데이트 내역

- **v2.0**: YOLO11 학습된 모델 지원 추가
- **v2.0**: 클래스 리스트 업데이트 (22개 클래스)
- **v2.0**: 인터랙티브 모드 및 시각적 편집 도구 추가
- **v2.0**: 통합 실행 스크립트 (`start_collection.sh`) 추가
- **v2.0**: 데이터셋 검토 및 편집 도구 추가 