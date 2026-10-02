"""
Wound Recovery Analysis 공통 모듈.

app.py와 Kaggle 노트북이 같은 로직(후처리 / 지표 / RVI / 패턴 분류 / 모델 구성)을
공유하도록 한 곳에 모아 둔 패키지.

- wound.config      : 공통 상수 (임계값, 입력 크기 등)
- wound.postprocess : 마스크 후처리
- wound.metrics     : IoU / Dice / Precision / Recall
- wound.analysis    : RVI, 패턴 분류, 안정화일 추정 (torch 불필요)
- wound.model       : Mask R-CNN 구성 / 로드 / 추론 (torch 필요)
"""
