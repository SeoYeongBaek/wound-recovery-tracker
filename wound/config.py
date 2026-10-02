"""공통 상수. 노트북과 앱이 같은 값을 쓰도록 여기서만 정의한다."""

# ===== MODEL =====
NUM_CLASSES   = 2      # 0: background, 1: wound
RESIZE_TO     = 512

# ===== INFERENCE =====
SCORE_THR     = 0.5
MASK_THR      = 0.5
MAX_INSTANCES = 5      # 이미지 한 장당 최대 검출 인스턴스 수

# ===== POSTPROCESS =====
K_CLOSE  = 7           # morphological closing 커널 크기
K_OPEN   = 3           # morphological opening 커널 크기
MIN_AREA = 200         # 이 픽셀 수 미만의 컴포넌트는 제거

# ===== RVI / 패턴 분류 =====
REF_DAYS    = 14       # RVI 정규화 기준일 (2주)
REBOUND_THR = 0.05     # 구간 면적 증가율이 이 값을 넘으면 rebound
PLATEAU_THR = 0.10     # 후반 3구간 변화율이 모두 이 값 미만이면 plateau
STABLE_RATIO = 0.05    # 면적이 초기 면적의 이 비율 이하가 되면 '안정화'
STABLE_HORIZON_DAYS = 180  # 안정화일 외삽 최대 탐색 범위 (마지막 측정일 기준)
