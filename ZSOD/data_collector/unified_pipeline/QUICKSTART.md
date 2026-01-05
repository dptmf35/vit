# Quick Start Guide - Unified ML Pipeline

## 빠른 시작 (5분 안에 시작하기)

### 1. 설치 확인

```bash
# Python 패키지 확인
python3 -c "import tkinter; import cv2; import ultralytics; print('✅ All packages installed')"

# ROS2 확인
ros2 topic list
```

### 2. GUI 실행

```bash
cd /home/yeseul/Desktop/mygitrepos/vit/ZSOD/data_collector/unified_pipeline
python3 main.py
```

### 3. 첫 번째 파이프라인 실행

#### Step 1: Configuration (⚙️ 탭)
1. Target Classes 입력: `chair, table, bed`
2. "💾 Save Configuration" 클릭

#### Step 2: Data Collection (📸 탭)
1. "🔍 Test Mode" 클릭 (먼저 테스트)
2. 카메라가 정상 작동하는지 확인
3. "⏹️ Stop Collection" 클릭
4. "▶️ Start Collection" 클릭 (실제 수집)
5. 30~100개 이미지 수집 후 중지

#### Step 3: Annotation Review (✏️ 탭)
1. "🔍 Launch Reviewer" 클릭
2. 어노테이션 검증 및 수정
3. 잘못된 데이터 삭제

#### Step 4: Training (🎓 탭)
1. "📊 Dataset Analysis" 클릭
2. "▶️ Start Training" 클릭
3. 학습 완료 대기 (10~30분)

#### Step 5: Evaluation (📊 탭)
1. "📊 Evaluate on Dataset" 클릭
2. 성능 메트릭 확인

#### Step 6: Deployment (🚀 탭)
1. Model Path 확인
2. "▶️ Start Deployment" 클릭
3. ROS2 토픽으로 결과 확인:
   ```bash
   ros2 topic echo /custom_yolo/detections
   ```

## 일반적인 사용 시나리오

### 시나리오 1: 빠른 프로토타입

**목표**: 최소한의 데이터로 빠르게 모델 생성

```
1. 20~30개 이미지 수집
2. 간단히 검증
3. 50 epochs로 빠른 학습
4. 배포 테스트
```

### 시나리오 2: 고품질 모델

**목표**: 높은 정확도의 프로덕션 모델

```
1. 200~500개 이미지 수집
2. 철저한 어노테이션 검증
3. 200 epochs로 완전 학습
4. 다양한 테스트 이미지로 평가
5. 프로덕션 배포
```

### 시나리오 3: 반복 개선

**목표**: 기존 모델을 개선

```
1. 기존 데이터셋 로드
2. 추가 데이터 수집
3. 재학습
4. 성능 비교
5. 배포 업데이트
```

## 팁과 트릭

### 데이터 수집
- ✅ **좋음**: Confidence threshold 0.6~0.8 (고품질)
- ❌ **피하기**: Threshold < 0.5 (저품질 데이터)

### 학습
- 소규모 데이터셋 (<50 images): yolo11n.pt, 50-100 epochs
- 중규모 데이터셋 (50-200 images): yolo11s.pt, 100-150 epochs
- 대규모 데이터셋 (>200 images): yolo11m.pt, 150-200 epochs

### 평가
- mAP50 > 0.7: 좋은 모델
- mAP50 > 0.5: 사용 가능한 모델
- mAP50 < 0.5: 더 많은 데이터 필요

## 문제 해결

### GUI가 열리지 않음
```bash
sudo apt-get install python3-tk
```

### ROS2 토픽이 안 보임
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
```

### 학습이 너무 느림
- GPU 확인: `nvidia-smi`
- 더 작은 모델 사용
- Epochs 줄이기

## 다음 단계

1. ✅ 기본 파이프라인 완료
2. 📚 README.md 참조하여 고급 기능 학습
3. 🔧 파라미터 튜닝으로 성능 최적화
4. 🚀 프로덕션 환경에 배포

## 도움말

- 전체 문서: `README.md`
- 설정 파일: `config/pipeline_config.json`
- 로그 파일: `logs/`

즐거운 ML 파이프라인 사용 되세요! 🎉
