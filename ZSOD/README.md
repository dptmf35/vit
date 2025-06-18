

data_collector dir과 train_model dir에 대해 설명

data collector dir -> isaac sim에서 yolo-e의 text prompting을 통한 초기 detection 데이터 수집 
(./start_collection.sh)

data_reviewer를 통해 수집한 데이터를 검증하고 편집한 후,

train_model dir에서 yolov11로 학습시킴

ZSOD의 custom_yolo_detector.py를 통해 학습결과 검증