# YOLO Dataset Reviewer

YOLO 형식 데이터셋을 검토하고 편집할 수 있는 GUI 도구입니다. 수집된 데이터의 품질을 확인하고 잘못된 라벨링을 수정하거나 불필요한 이미지를 삭제할 수 있습니다.

## 기능

### 주요 기능
- **이미지 탐색**: 데이터셋 이미지를 하나씩 순서대로 검토
- **어노테이션 시각화**: YOLO 포맷 라벨을 바운딩 박스로 시각화
- **파일 삭제**: 잘못된 데이터 완전 삭제 (이미지 + 라벨 + 시각화 파일)
- **라벨 편집**: 텍스트 에디터로 어노테이션 직접 수정
- **통계 정보**: 데이터셋 상태 및 작업 진행률 확인

### 지원 형식
- **이미지**: JPG, JPEG, PNG, BMP
- **라벨**: YOLO 포맷 (.txt)
- **클래스**: dataset.yaml에서 자동 로드 또는 기본 클래스 사용

## 설치 및 요구사항

### 필수 패키지
```bash
# OpenCV
pip install opencv-python

# PIL (Pillow)
pip install pillow

# tkinter (usually pre-installed with Python)
sudo apt-get install python3-tk  # Ubuntu/Debian
```

## 사용법

### 1. 기본 실행
```bash
# 런처 스크립트 사용 (권장)
python3 run_dataset_reviewer.py

# 직접 실행
python3 dataset_reviewer.py /path/to/dataset
```

### 2. 옵션 설정
```bash
# 특정 데이터셋 디렉토리 지정
python3 run_dataset_reviewer.py --dataset-dir /path/to/your/dataset

# 특정 이미지 인덱스부터 시작
python3 run_dataset_reviewer.py --start-index 50

# 도움말
python3 run_dataset_reviewer.py --help
```

## 데이터셋 구조

```
dataset_directory/
├── images/           # 이미지 파일들
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── labels/           # YOLO 포맷 라벨 파일들
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── visualizations/   # 시각화 파일들 (선택사항)
│   ├── vis_image1.jpg
│   └── ...
└── dataset.yaml      # 클래스 정의 (선택사항)
```

## 인터페이스 가이드

### 메인 화면
```
┌─────────────────────────────────────────────────────────────┐
│ Image 1/100: image001.jpg          Deleted: 0 | Edited: 0   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                 [이미지 표시 영역]                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Annotations:                                                │
│ 1. chair (ID: 3) - (100, 200, 300, 400)                   │
│ 2. table (ID: 12) - (150, 250, 350, 450)                  │
├─────────────────────────────────────────────────────────────┤
│ [◀◀ First] [◀ Previous] [Next ▶] [Last ▶▶]               │
│                    [🗑️ Delete] [✏️ Edit] [🔄 Refresh] [📊 Stats] │
└─────────────────────────────────────────────────────────────┘
```

### 키보드 단축키
- **탐색**:
  - `←` / `A`: 이전 이미지
  - `→` / `D`: 다음 이미지
  - `Home`: 첫 번째 이미지
  - `End`: 마지막 이미지

- **작업**:
  - `Delete`: 현재 이미지 삭제
  - `E`: 라벨 편집
  - `R`: 새로고침
  - `Esc`: 종료

### 라벨 편집기
```
┌─────────────────────────────────────────────────────────────┐
│ Edit Labels - image001.txt                                  │
├─────────────────────────────────────────────────────────────┤
│ 3 0.5 0.6 0.2 0.3                                         │
│ 12 0.7 0.4 0.15 0.25                                      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Format: class_id center_x center_y width height (0-1)      │
│ Classes: 0:bed, 1:cabinet, 2:carpet, 3:chair, ...         │
├─────────────────────────────────────────────────────────────┤
│                           [💾 Save] [❌ Cancel]              │
└─────────────────────────────────────────────────────────────┘
```

## 기능 상세

### 1. 이미지 삭제
- 현재 이미지와 관련된 모든 파일 삭제:
  - 원본 이미지 파일
  - 어노테이션 텍스트 파일
  - 시각화 이미지 파일 (있는 경우)
- 삭제 전 확인 대화상자 표시
- 되돌릴 수 없는 작업이므로 주의 필요

### 2. 라벨 편집
- YOLO 형식 검증:
  - 5개 값 필수: `class_id center_x center_y width height`
  - 좌표값은 0~1 사이 정규화된 값
  - class_id는 유효한 범위 내 정수
- 실시간 형식 검증
- 저장 시 자동으로 이미지 새로고침

### 3. 클래스 관리
데이터셋의 클래스 정보는 다음 순서로 로드됩니다:
1. `dataset.yaml` 파일의 `names` 항목
2. 기본 클래스 목록 (가구/생활용품)

기본 클래스:
```python
["bed", "cabinet", "carpet", "chair", "closet", "curtain", 
 "dish", "door", "fridge", "gas stove", "hanger", "lamp", 
 "microwave", "nightstand", "plant", "shelf", "sofa", 
 "table", "tv", "window", "vanity"]
```

### 4. 통계 정보
- 전체 이미지 수
- 현재 위치
- 삭제된 이미지 수
- 편집된 라벨 수
- 사용 가능한 클래스 목록

## 작업 흐름 예시

### 데이터셋 검토 작업
1. **시작**: `python3 run_dataset_reviewer.py`
2. **검토**: 각 이미지의 어노테이션 품질 확인
3. **수정**: 잘못된 라벨은 `E` 키로 편집
4. **삭제**: 품질이 나쁜 이미지는 `Delete` 키로 삭제
5. **진행**: `→` 키로 다음 이미지로 이동
6. **완료**: 모든 이미지 검토 후 프로그램 종료

### 특정 범위 검토
```bash
# 50번째 이미지부터 시작
python3 run_dataset_reviewer.py --start-index 50

# 특정 데이터셋 검토
python3 run_dataset_reviewer.py --dataset-dir /home/user/custom_dataset
```

## 주의사항

### 데이터 안전
- **삭제는 영구적**: 삭제된 파일은 복구할 수 없습니다
- **백업 권장**: 중요한 데이터는 미리 백업하세요
- **검증**: 편집한 라벨은 저장 전 형식이 검증됩니다

### 성능
- 대용량 이미지는 800x600으로 자동 리사이즈됩니다
- 메모리 사용량 최적화를 위해 한 번에 하나의 이미지만 로드합니다

### 호환성
- Python 3.6 이상 필요
- tkinter GUI 지원 환경 필요
- OpenCV 및 PIL 라이브러리 필요

## 문제 해결

### 일반적인 오류
1. **"Images directory not found"**
   - 데이터셋 디렉토리 경로 확인
   - `images/` 폴더 존재 여부 확인

2. **"No images found"**
   - 지원되는 이미지 형식 확인 (jpg, png, bmp)
   - 파일 권한 확인

3. **"Failed to load image"**
   - 이미지 파일 손상 여부 확인
   - 파일 형식 재확인

### GUI 문제
- **화면이 너무 작음**: 최소 1200x800 해상도 권장
- **한글 폰트 문제**: 시스템 폰트 설정 확인
- **tkinter 오류**: `sudo apt-get install python3-tk` 설치

## 개발자 정보

이 도구는 YOLO 데이터셋 수집 시스템의 일부로 개발되었습니다.
- 메인 수집 도구: `yolo_dataset_collector.py`
- 시각적 프롬프트: `visual_prompt_detector.py`
- 데이터셋 검토: `dataset_reviewer.py` (현재 도구)

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 개발되었습니다. 