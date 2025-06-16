# YOLO Dataset Collector

ROS2 텍스트 프롬프트 기반 YOLO 데이터셋 자동 수집 도구입니다.

## 주요 기능

- **자동 데이터셋 수집**: 특정 객체가 인식될 때마다 자동으로 이미지와 어노테이션 저장
- **YOLO 형식 지원**: 표준 YOLO 학습 형식으로 데이터 저장 (normalized coordinates)
- **적응형 수집**: confidence threshold, IOU threshold, 수집 간격 등 세밀한 제어
- **통계 기능**: 실시간 수집 통계 및 클래스별 수집 현황 모니터링
- **텍스트 프롬프트**: 특정 객체들을 텍스트로 지정하여 학습 데이터 수집

## 지원하는 객체 클래스

```
"table", "fridge", "chair", "dish", "gas stove", "closet", 
"lamp", "curtain", "nightstand", "microwave", "tv", "sofa", 
"shelf", "window", "door", "bed", "cabinet", "plant", "computer"
```

## 파일 구조

```
├── yolo_dataset_collector.py    # 메인 데이터셋 수집기
├── run_dataset_collector.py     # 설정 가능한 런처 스크립트
└── DATASET_COLLECTOR_README.md  # 사용법 안내
```

## 설치 및 의존성

```bash
# ROS2 및 필요한 패키지들이 설치되어 있어야 합니다
pip install ultralytics opencv-python
```

## 사용법

### 1. 기본 실행
```bash
# 기본 설정으로 실행
python3 run_dataset_collector.py
```

### 2. 고급 설정으로 실행
```bash
# 높은 품질의 데이터 수집 (높은 confidence threshold)
python3 run_dataset_collector.py --conf_threshold 0.8 --collection_interval 3.0

# 빠른 수집 (낮은 threshold, 짧은 간격)
python3 run_dataset_collector.py --conf_threshold 0.5 --collection_interval 1.0

# 다른 이미지 토픽 사용
python3 run_dataset_collector.py --image_topic /camera/image_raw

# 커스텀 저장 경로 설정
python3 run_dataset_collector.py --dataset_path ~/my_custom_dataset
```

### 3. 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--conf_threshold` | 0.6 | 객체 인식 confidence threshold (0.0-1.0) |
| `--iou_threshold` | 0.4 | Non-Maximum Suppression IOU threshold |
| `--collection_interval` | 2.0 | 데이터 수집 간격 (초) |
| `--min_detections` | 1 | 저장하기 위한 최소 detection 수 |
| `--max_detections` | 50 | 이미지당 최대 detection 수 |
| `--dataset_path` | ~/yolo_dataset | 데이터셋 저장 경로 |
| `--image_topic` | /stereo_image_color | 입력 이미지 ROS 토픽 |
| `--model_path` | yoloe-11s-seg.pt | YOLOE 모델 파일 경로 |

## 생성되는 데이터셋 구조

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
[INFO] YOLO Dataset Collector initialized
[INFO] Collection parameters: conf=0.6, iou=0.4
[INFO] Collection interval: 2.0s
[INFO] Saved dataset sample 1: img_20241220_143025_123.jpg
[INFO] Collected data with classes: table, chair
[INFO] === Collection Statistics (Total: 10) ===
[INFO]   table: 6
[INFO]   chair: 4
[INFO]   tv: 2
```

## 최적화 팁

### 고품질 데이터셋을 위한 설정
```bash
python3 run_dataset_collector.py \
    --conf_threshold 0.8 \
    --iou_threshold 0.3 \
    --collection_interval 3.0 \
    --min_detections 2
```

### 대량 수집을 위한 설정
```bash
python3 run_dataset_collector.py \
    --conf_threshold 0.5 \
    --collection_interval 1.0 \
    --max_detections 100
```

### 특정 객체 집중 수집
특정 객체가 많이 나오는 환경에서 실행하여 해당 클래스의 데이터를 집중적으로 수집할 수 있습니다.

## 주의사항

1. **모델 파일**: `yoloe-11s-seg.pt` 파일이 실행 디렉토리에 있어야 합니다.
2. **ROS2 토픽**: 지정한 이미지 토픽이 활성화되어 있어야 합니다.
3. **저장 공간**: 대량 수집 시 충분한 디스크 공간을 확보하세요.
4. **수집 품질**: threshold를 너무 낮게 설정하면 저품질 데이터가 수집될 수 있습니다.

## 수집 중단 및 재개

- `Ctrl+C`로 안전하게 중단할 수 있습니다.
- 중단 시 마지막 통계가 출력됩니다.
- 재시작하면 기존 데이터셋에 추가로 수집됩니다 (덮어쓰지 않음).

## 트러블슈팅

### 일반적인 문제들

1. **모델 로딩 실패**
   ```
   ERROR: Cannot load model yoloe-11s-seg.pt
   ```
   → 모델 파일 경로를 확인하세요.

2. **토픽 수신 실패**
   ```
   No images received on topic /stereo_image_color
   ```
   → `ros2 topic list`로 토픽 이름을 확인하세요.

3. **권한 오류**
   ```
   Permission denied: ~/yolo_dataset
   ```
   → 저장 경로의 쓰기 권한을 확인하세요.

## YOLO 모델 학습

수집된 데이터셋은 바로 YOLO 학습에 사용할 수 있습니다:

```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('yolov8n.pt')

# 학습 실행
results = model.train(data='~/yolo_dataset/dataset.yaml', epochs=100)
``` 