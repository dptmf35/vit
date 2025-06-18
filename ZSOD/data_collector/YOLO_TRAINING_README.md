# YOLO Training for Small Datasets

작은 데이터셋으로 YOLO11 모델을 학습시키는 도구입니다. 50개 미만의 이미지로도 효과적인 transfer learning을 통해 학습이 가능합니다.

## 🚀 빠른 시작

### 1. 데이터셋 분석
```bash
# 데이터셋 상태 확인
python3 run_yolo_training.py --analyze-only

# 또는 직접 실행
python3 train_yolo.py --dataset ../yolo_dataset --analyze-only
```

### 2. 학습 시작
```bash
# 기본 설정으로 학습
python3 run_yolo_training.py

# 빠른 테스트 (20 에포크)
python3 run_yolo_training.py --quick-train

# 커스텀 설정
python3 run_yolo_training.py --model yolo11s.pt --epochs 150 --batch-size 4
```

### 3. 학습된 모델 테스트
```bash
# 단일 이미지 테스트
python3 test_trained_model.py training_output/train/weights/best.pt --input test_image.jpg

# 디렉토리 전체 테스트
python3 test_trained_model.py training_output/train/weights/best.pt --input test_images/

# 인터랙티브 모드
python3 test_trained_model.py training_output/train/weights/best.pt --interactive
```

## 📋 요구사항

### 필수 패키지
```bash
# YOLO
pip install ultralytics

# 추가 패키지
pip install opencv-python
pip install PyYAML
pip install numpy
```

### 데이터셋 구조
```
yolo_dataset/
├── images/           # 이미지 파일들
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── labels/           # YOLO 형식 라벨
│   ├── img_001.txt
│   ├── img_002.txt
│   └── ...
└── dataset.yaml      # 클래스 정의
```

## 🔧 사용법 상세

### 기본 학습
```bash
# 런처 스크립트 사용 (추천)
python3 run_yolo_training.py

# 직접 실행
python3 train_yolo.py --dataset /path/to/dataset --model yolo11s.pt --epochs 100
```

### 모델 선택
- `yolo11n.pt`: 가장 작고 빠름 (2.6M 파라미터)
- `yolo11s.pt`: 작은 크기, 균형잡힌 성능 (9.4M) **[추천]**
- `yolo11m.pt`: 중간 크기 (20.1M)
- `yolo11l.pt`: 큰 크기 (25.3M)
- `yolo11x.pt`: 가장 큰 크기 (68.2M)

```bash
# 작은 데이터셋에는 yolo11s.pt 추천
python3 run_yolo_training.py --model yolo11s.pt

# 매우 작은 데이터셋에는 yolo11n.pt
python3 run_yolo_training.py --model yolo11n.pt --epochs 200
```

### 배치 크기 자동 조정
- 10개 미만 이미지: batch_size = 2
- 30개 미만 이미지: batch_size = 4  
- 30개 이상 이미지: batch_size = 8

```bash
# 수동 설정
python3 run_yolo_training.py --batch-size 2
```

### 에포크 수 권장사항
- **10개 미만**: 150-300 에포크
- **10-30개**: 100-200 에포크
- **30개 이상**: 50-150 에포크

```bash
# 작은 데이터셋용 장시간 학습
python3 run_yolo_training.py --epochs 200

# 빠른 테스트
python3 run_yolo_training.py --quick-train  # 20 에포크
```

## 📊 소규모 데이터셋 최적화

### 자동 최적화 기능
- **Transfer Learning**: 사전 훈련된 가중치 사용
- **낮은 학습률**: 과적합 방지 (lr0=0.001)
- **보수적 데이터 증강**: 작은 데이터셋에 적합
- **조기 종료**: patience 설정으로 과적합 방지
- **스마트 분할**: 최소 검증 데이터 보장

### 데이터 증강 설정
```python
# 작은 데이터셋에 최적화된 증강
'hsv_h': 0.01,      # 색조 변화 최소화
'hsv_s': 0.5,       # 채도 변화 중간
'hsv_v': 0.3,       # 명도 변화 중간
'degrees': 0.0,     # 회전 없음
'translate': 0.1,   # 약간의 이동
'scale': 0.2,       # 약간의 크기 변화
'fliplr': 0.5,      # 좌우 반전만
'mosaic': 0.3,      # 모자이크 감소
'mixup': 0.0,       # 믹스업 비활성화
```

## 🎯 성능 향상 팁

### 데이터 품질 개선
1. **라벨 품질 확인**: Dataset Reviewer 사용
2. **클래스 균형**: 각 클래스당 최소 10개 이상
3. **다양성 확보**: 다양한 각도, 조명, 배경

### 학습 설정 조정
```bash
# GPU 메모리가 부족한 경우
python3 run_yolo_training.py --batch-size 1 --imgsz 416

# 더 정확한 모델이 필요한 경우
python3 run_yolo_training.py --model yolo11m.pt --epochs 200

# 빠른 추론이 필요한 경우
python3 run_yolo_training.py --model yolo11n.pt --imgsz 320
```

## 📈 결과 분석

### 학습 결과 확인
```
training_output/
├── train/
│   ├── weights/
│   │   ├── best.pt      # 최고 성능 모델
│   │   └── last.pt      # 마지막 에포크 모델
│   ├── results.png      # 학습 곡선
│   ├── confusion_matrix.png
│   ├── F1_curve.png
│   └── PR_curve.png
└── dataset.yaml         # 학습용 데이터셋 설정
```

### 모델 성능 평가
```bash
# 검증 데이터로 평가
python3 -c "
from ultralytics import YOLO
model = YOLO('training_output/train/weights/best.pt')
results = model.val()
print(f'mAP50: {results.box.map50:.3f}')
print(f'mAP50-95: {results.box.map:.3f}')
"
```

## 🧪 모델 테스트

### 단일 이미지 테스트
```bash
python3 test_trained_model.py training_output/train/weights/best.pt --input test.jpg --conf 0.3
```

### 배치 테스트
```bash
python3 test_trained_model.py training_output/train/weights/best.pt --input test_images/ --conf 0.25
```

### 인터랙티브 테스트
```bash
python3 test_trained_model.py training_output/train/weights/best.pt --interactive
```

## 🚨 문제 해결

### 일반적인 문제들

#### 1. "CUDA out of memory"
```bash
# 배치 크기 줄이기
python3 run_yolo_training.py --batch-size 1

# 이미지 크기 줄이기
python3 run_yolo_training.py --imgsz 416
```

#### 2. "No objects detected"
```bash
# confidence threshold 낮추기
python3 test_trained_model.py model.pt --input image.jpg --conf 0.1

# 더 많은 에포크로 재학습
python3 run_yolo_training.py --epochs 300
```

#### 3. "Overfitting detected"
- 더 적은 에포크 사용
- 데이터 증강 강화
- 더 작은 모델 사용 (yolo11n.pt)

#### 4. "Dataset too small"
```bash
# 데이터 분석으로 상태 확인
python3 run_yolo_training.py --analyze-only

# 권장사항:
# - 각 클래스당 최소 10개 이상
# - 전체 50개 이상 이미지
# - 균형잡힌 클래스 분포
```

## 📚 고급 사용법

### 커스텀 클래스 사용
```python
# dataset.yaml 수정
names: ['my_class1', 'my_class2', 'my_class3']
nc: 3
```

### 하이퍼파라미터 튜닝
```bash
# 학습률 조정
python3 train_yolo.py --dataset ../yolo_dataset --lr0 0.0005

# 가중치 감쇠 조정  
python3 train_yolo.py --dataset ../yolo_dataset --weight_decay 0.001
```

### 체크포인트에서 재시작
```python
from ultralytics import YOLO

# 중단된 학습 재시작
model = YOLO('training_output/train/weights/last.pt')
model.train(resume=True)
```

## 💡 작은 데이터셋 성공 사례

### 권장 워크플로우
1. **데이터 수집**: 최소 20-50개 이미지
2. **라벨링**: Dataset Reviewer로 정확한 라벨링
3. **분석**: `--analyze-only`로 데이터 품질 확인
4. **학습**: yolo11s.pt로 150-200 에포크
5. **평가**: 테스트 이미지로 성능 확인
6. **반복**: 부족한 클래스 데이터 추가 수집

### 성공 지표
- **mAP50 > 0.3**: 기본적인 검출 가능
- **mAP50 > 0.5**: 실용적인 성능
- **mAP50 > 0.7**: 우수한 성능

## 📝 요약

50개 미만의 작은 데이터셋으로도 YOLO 학습이 가능합니다:

✅ **가능한 것들**:
- 실용적인 객체 검출 모델 학습
- Transfer learning을 통한 빠른 수렴
- 특정 도메인에 특화된 모델 생성

⚠️ **주의사항**:
- 일반화 성능은 제한적
- 클래스별 충분한 예시 필요
- 과적합 주의 깊게 모니터링

🎯 **성공 요인**:
- 고품질 라벨링
- 적절한 하이퍼파라미터
- 충분한 학습 시간 (에포크)
- 균형잡힌 데이터 분포 