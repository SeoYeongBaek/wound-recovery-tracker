"""
Wound Recovery Analysis — Streamlit Demo App
이미지를 누적 업로드하면 안정화일을 예측하는 웹 데모

시작 시, 터미널에...
streamlit run app.py
"""

import os
import platform

import cv2
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision
from PIL import Image
from scipy.optimize import curve_fit
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    os.path.join(_SCRIPT_DIR, "mask_rcnn_wound_final.pth"),           # 프로젝트 폴더
    os.path.expanduser("~/Downloads/mask_rcnn_wound_final.pth"),       # 다운로드 폴더
    os.path.expanduser("~/mask_rcnn_wound_final.pth"),                  # 홈 폴더
]
MODEL_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])
NUM_CLASSES = 2
RESIZE_TO   = 512
SCORE_THR     = 0.5
MASK_THR      = 0.5
MAX_INSTANCES = 5   # 이미지 한 장당 최대 검출 인스턴스 수

# 인스턴스별 오버레이 색상 (BGR → RGB 변환 후 사용)
INSTANCE_COLORS_RGB = [
    (220,  40,  40),   # #1 빨강
    ( 40,  80, 220),   # #2 파랑
    ( 40, 180,  40),   # #3 초록
    (200, 180,   0),   # #4 노랑
    (180,  40, 180),   # #5 마젠타
]

REBOUND_THR = 0.05
PLATEAU_THR = 0.10
STABLE_THR  = 0.05
REF_DAYS    = 14

PATTERN_COLORS = {
    "normal":  "#2ecc71",
    "plateau": "#3498db",
    "rebound": "#e74c3c",
    "unknown": "#95a5a6",
}
PATTERN_KR = {
    "normal":  "정상 회복",
    "plateau": "회복 정체",
    "rebound": "일시 악화",
    "unknown": "데이터 부족",
}

# ─────────────────────────────────────────────
#  Korean font setup (회복 곡선 글자 깨짐 방지)
# ─────────────────────────────────────────────
def _setup_korean_font():
    """macOS / Windows / Linux 환경에서 한글 폰트 자동 설정."""
    candidates = {
        "Darwin":  ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"],
        "Windows": ["Malgun Gothic", "NanumGothic"],
        "Linux":   ["NanumGothic", "NanumBarunGothic", "UnDotum"],
    }.get(platform.system(), ["NanumGothic"])
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_korean_font()

# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="모델 로딩 중...")
def load_model(model_path: str):
    """학습된 Mask R-CNN 모델을 로드하고 반환."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


# ─────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────
def postprocess_mask(bin_mask: np.ndarray,
                     k_close: int = 7, k_open: int = 3,
                     min_area: int = 200) -> np.ndarray:
    """Morphological closing/opening + 소형 컴포넌트 제거."""
    m = (bin_mask * 255).astype(np.uint8)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    m  = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    m  = cv2.morphologyEx(m, cv2.MORPH_OPEN,  ko)
    m  = (m > 0).astype(np.uint8)
    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = np.zeros_like(m)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 1
        m = out
    return m


def run_inference(model, device, pil_image: Image.Image):
    """
    PIL 이미지에 Mask R-CNN 추론을 실행. 인스턴스별로 분리하여 반환.

    Mask R-CNN은 각 상처에 고유 ID를 부여하므로 다중 상처를 독립적으로 추적 가능.
    (U-Net 방식은 모든 상처를 단일 마스크로 합쳐 인스턴스 구분 불가)

    Returns:
        pred_bin (np.ndarray): 전체 예측 binary 마스크 (인스턴스 union, H×W)
        overlay_rgb (np.ndarray): 인스턴스별 색상 오버레이 (H×W×3, RGB)
        wound_area (int): 전체 상처 면적 (픽셀 수)
        num_instances (int): 검출된 인스턴스 수
        instance_areas (list[int]): 인스턴스별 면적 리스트
    """
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)

    with torch.inference_mode():
        out = model([img_t])[0]

    scores_np  = out.get("scores", torch.tensor([])).cpu().numpy()
    masks_tens = out.get("masks", None)

    # ── 인스턴스별 마스크 분리 수집 ──
    instance_masks = []
    instance_areas = []

    if len(scores_np) > 0 and masks_tens is not None:
        keep_idx = np.where(scores_np >= SCORE_THR)[0][:MAX_INSTANCES]
        for ki in keep_idx:
            m = (masks_tens[ki, 0].cpu().numpy() > MASK_THR).astype(np.uint8)
            m = postprocess_mask(m)
            if m.sum() > 0:
                instance_masks.append(m)
                instance_areas.append(int(m.sum()))

    # 전체 마스크: 인스턴스 union
    if instance_masks:
        pred_bin = np.stack(instance_masks).max(axis=0)
    else:
        pred_bin = np.zeros((RESIZE_TO, RESIZE_TO), dtype=np.uint8)

    # ── 인스턴스별 색상 오버레이 ──
    overlay = rgb.copy().astype(np.float32)
    for i, mask in enumerate(instance_masks):
        color = np.array(INSTANCE_COLORS_RGB[i % len(INSTANCE_COLORS_RGB)], dtype=np.float32)
        region = mask.astype(bool)
        overlay[region] = overlay[region] * 0.4 + color * 0.6
        # 인스턴스 번호 레이블
        ys, xs = np.where(region)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(overlay.astype(np.uint8), f"#{i+1}", (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    overlay_rgb = np.clip(overlay, 0, 255).astype(np.uint8)
    wound_area  = int(pred_bin.sum())
    return pred_bin, overlay_rgb, wound_area, len(instance_masks), instance_areas


# ─────────────────────────────────────────────
#  Analysis helpers
# ─────────────────────────────────────────────
def classify_pattern(areas: list) -> str:
    """
    시계열 면적으로 치유 패턴 분류 (Normal / Plateau / Rebound).
    최소 3개 데이터 포인트 필요.
    """
    if len(areas) < 3:
        return "unknown"
    n = len(areas)
    # Rebound: 어느 구간이든 면적 5% 이상 증가 (area=0인 구간은 직전 값 기준)
    for i in range(0, n - 1):
        ref = areas[i] if areas[i] > 0 else (areas[i - 1] if i > 0 else 0)
        if ref > 0 and (areas[i + 1] - areas[i]) / ref > REBOUND_THR:
            return "rebound"
    # Plateau: 후반 min(3, n-1)개 구간 변화율 모두 10% 미만
    late_idx_start = max(0, n - 4)
    late_changes = []
    for i in range(late_idx_start, n - 1):
        if areas[i] > 0:
            late_changes.append(abs(areas[i + 1] - areas[i]) / areas[i])
    if late_changes and all(c < PLATEAU_THR for c in late_changes):
        return "plateau"
    return "normal"


def compute_rvi(areas: list, days: list) -> float:
    """RVI(Recovery Velocity Index, 0~100) 계산."""
    if len(areas) < 2 or areas[0] <= 0:
        return 0.0
    a0, a_last = areas[0], areas[-1]
    t_elapsed = days[-1] - days[0]
    if t_elapsed <= 0:
        return 0.0
    total_reduction = (a0 - a_last) / a0
    rvi = total_reduction / t_elapsed * REF_DAYS * 100
    return float(np.clip(rvi, 0, 100))


def estimate_stable_day(areas: list, days: list) -> tuple[int, str]:
    """
    안정화 예상일 추정.

    Returns:
        (day_estimate, method_label)
        method_label: 'observed' | 'extrapolated' | 'unknown'
    """
    if len(areas) < 2:
        return days[-1], "unknown"

    # 이미 안정화된 경우: 마지막 변화율 < STABLE_THR
    for i in range(len(areas) - 1):
        if areas[i] > 0 and abs(areas[i + 1] - areas[i]) / areas[i] < STABLE_THR:
            return days[i + 1], "observed"

    # 지수 감소 피팅으로 외삽
    try:
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * t) + c

        a0 = float(areas[0])
        p0 = [a0, 0.05, a0 * 0.05]
        popt, _ = curve_fit(
            exp_decay, days, areas,
            p0=p0, maxfev=5000,
            bounds=([0, 1e-6, 0], [a0 * 10, 2, a0]),
        )
        # 안정화 기준: 초기 면적의 5% 이하
        target = a0 * 0.05
        # day를 연속적으로 탐색
        for d in range(days[-1], days[-1] + 180):
            if exp_decay(d, *popt) <= target:
                return d, "extrapolated"
        return days[-1] + 180, "extrapolated"

    except Exception:
        # 선형 외삽 fallback
        if len(areas) >= 2:
            rate = (areas[-1] - areas[-2]) / max(1, days[-1] - days[-2])
            if rate < 0:
                days_to_stable = (areas[-1] - areas[0] * 0.05) / (-rate)
                return int(days[-1] + max(0, days_to_stable)), "extrapolated"
        return days[-1], "unknown"


# ─────────────────────────────────────────────
#  Instance series helper
# ─────────────────────────────────────────────
def get_instance_series(records_sorted: list) -> list:
    """
    인스턴스별 시계열 면적 추출 (index 기반 매칭).

    Returns:
        list of {"days": list[int], "areas": list[int]} — 인스턴스 수만큼
    """
    n_inst = max((len(r.get("instance_areas", [])) for r in records_sorted), default=0)
    n_inst = min(n_inst, MAX_INSTANCES)
    result = []
    for idx in range(n_inst):
        days, areas = [], []
        for r in records_sorted:
            inst_areas = r.get("instance_areas", [])
            days.append(r["day"])
            areas.append(inst_areas[idx] if idx < len(inst_areas) else 0)
        result.append({"days": days, "areas": areas})
    return result


# ─────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────
def make_recovery_curve(records: list, stable_day: int, stable_method: str,
                        pattern: str) -> plt.Figure:
    """
    회복 곡선 matplotlib Figure 생성.
    records: list of {"day": int, "area": int}
    """
    days  = [r["day"]  for r in records]
    areas = [r["area"] for r in records]
    a0    = areas[0] if areas[0] > 0 else 1
    norm  = [a / a0 for a in areas]

    color = PATTERN_COLORS.get(pattern, "#95a5a6")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(days, norm, marker="o", color=color, linewidth=2.5,
            markersize=7, label="측정값")

    # 안정화 기준선
    ax.axhline(0.05, color="#7f8c8d", linestyle=":", linewidth=1.2,
               label="안정화 기준 (5%)")

    # 안정화 예상일 수직선
    if stable_method != "unknown":
        ax.axvline(stable_day, color="#e67e22", linestyle="--", linewidth=1.5,
                   label=f"안정화 예상: Day {stable_day}"
                         + (" (관측)" if stable_method == "observed" else " (추정)"))

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("정규화 면적 (Day 0 기준)", fontsize=12)
    ax.set_title(f"회복 곡선 — {PATTERN_KR.get(pattern, '?')}", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Session state init
# ─────────────────────────────────────────────
def init_session():
    if "records" not in st.session_state:
        st.session_state.records = []   # list of {"day", "area", "overlay"}
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


# ─────────────────────────────────────────────
#  Main App
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Wound Recovery Analysis",
        page_icon="🩹",
        layout="wide",
    )

    init_session()

    # ── Header ──────────────────────────────
    st.title("🩹 Wound Recovery Analysis")
    st.caption(
        "상처 이미지를 날짜 순으로 업로드하면, "
        "Mask R-CNN이 상처 면적을 분석하고 **예상 안정화일**을 알려줍니다."
    )
    st.divider()

    # ── 모델 경로 확인 ───────────────────────
    active_model_path = MODEL_PATH
    if not os.path.exists(active_model_path):
        st.error(
            f"모델 파일을 찾을 수 없습니다: `{active_model_path}`\n\n"
            "Kaggle에서 다운로드한 `mask_rcnn_wound_final.pth`를 "
            "이 스크립트와 같은 폴더에 넣거나, 사이드바에서 직접 업로드하세요."
        )
        st.stop()

    model, device = load_model(active_model_path)

    # ── Sidebar: 이미지 업로드 ───────────────
    with st.sidebar:
        st.header("📷 이미지 업로드")

        uploaded = st.file_uploader(
            "상처 이미지 선택",
            type=["jpg", "jpeg", "png"],
            key="uploader",
        )
        day_input = st.number_input(
            "촬영일 (Day)", min_value=0, max_value=365, value=0, step=1,
        )
        analyze_btn = st.button("분석 추가", type="primary", use_container_width=True)

        st.divider()
        if st.button("초기화 (새 케이스)", use_container_width=True):
            st.session_state.records = []
            st.rerun()

        # 누적 데이터 요약
        if st.session_state.records:
            st.subheader("누적 데이터")
            for r in sorted(st.session_state.records, key=lambda x: x["day"]):
                st.write(f"Day {r['day']:>3d} → {r['area']:,} px")

    # ── 분석 실행 ────────────────────────────
    if analyze_btn and uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")

        with st.spinner("상처 면적 분석 중..."):
            _, overlay_rgb, wound_area, n_inst, inst_areas = run_inference(model, device, pil_img)

        if n_inst >= 2:
            st.sidebar.success(f"✅ {n_inst}개 상처 인스턴스 분리 검출")
        elif n_inst == 1:
            st.sidebar.info("상처 1개 검출")
        else:
            st.sidebar.warning("상처가 검출되지 않았습니다.")

        # 중복 day 처리 (덮어쓰기)
        st.session_state.records = [
            r for r in st.session_state.records if r["day"] != day_input
        ]
        st.session_state.records.append({
            "day":          day_input,
            "area":         wound_area,
            "overlay":      overlay_rgb,
            "num_instances": n_inst,
            "instance_areas": inst_areas,
        })
        st.rerun()

    elif analyze_btn and uploaded is None:
        st.sidebar.warning("이미지를 먼저 선택해주세요.")

    # ── 결과 표시 ────────────────────────────
    records_sorted = sorted(st.session_state.records, key=lambda x: x["day"])

    if not records_sorted:
        st.info(
            "왼쪽 사이드바에서 상처 이미지를 업로드하고 촬영일을 입력한 뒤 "
            "**[분석 추가]** 버튼을 클릭하세요.\n\n"
            "📌 여러 날짜의 이미지를 누적 업로드할수록 예측 정확도가 높아집니다."
        )
        return

    # ── 최신 오버레이 이미지 + 핵심 지표 ─────
    latest = records_sorted[-1]
    days  = [r["day"]  for r in records_sorted]
    areas = [r["area"] for r in records_sorted]

    col_img, col_metrics = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader(f"최신 상처 마스크 (Day {latest['day']})")
        st.image(latest["overlay"], use_container_width=True, clamp=True)
        n_inst = latest.get("num_instances", 1)
        inst_areas = latest.get("instance_areas", [latest["area"]])
        if n_inst >= 2:
            areas_str = " / ".join([f"#{i+1}: {a:,} px" for i, a in enumerate(inst_areas)])
            st.caption(f"**{n_inst}개 상처 인스턴스 분리 검출** — {areas_str}")
        else:
            st.caption(f"감지된 상처 면적: **{latest['area']:,} px**")

    with col_metrics:
        st.subheader("회복 분석 결과")

        pattern = classify_pattern(areas)
        rvi     = compute_rvi(areas, days)

        if len(records_sorted) >= 2:
            stable_day, stable_method = estimate_stable_day(areas, days)
        else:
            stable_day, stable_method = days[-1], "unknown"

        # 패턴 배지
        badge_color = PATTERN_COLORS.get(pattern, "#95a5a6")
        st.markdown(
            f"<div style='background:{badge_color};color:white;padding:8px 16px;"
            f"border-radius:8px;display:inline-block;font-weight:bold;font-size:16px;'>"
            f"패턴: {PATTERN_KR.get(pattern, '?')}</div>",
            unsafe_allow_html=True,
        )
        st.write("")

        m1, m2 = st.columns(2)
        m1.metric("RVI (회복 속도 지수)", f"{rvi:.1f} / 100")
        m2.metric("데이터 포인트", f"{len(records_sorted)}개")

        if stable_method != "unknown":
            label = "✅ 관측" if stable_method == "observed" else "📊 추정"
            st.success(
                f"**예상 안정화일: Day {stable_day}** {label}\n\n"
                f"(초기 면적의 5% 이하로 감소하는 시점)"
            )
        else:
            st.warning(
                "이미지를 더 업로드하면 안정화일을 추정할 수 있습니다.\n"
                "(최소 2개 이상의 날짜 데이터 필요)"
            )

        # 패턴별 설명
        descriptions = {
            "normal":  "상처가 꾸준히 감소하고 있습니다. 정상적인 회복 경과입니다.",
            "plateau": "최근 상처 크기 변화가 거의 없습니다. 회복이 정체된 상태입니다.",
            "rebound": "중간에 상처가 일시적으로 커졌다가 다시 회복 중입니다.",
            "unknown": "아직 데이터가 부족합니다. 이미지를 추가로 업로드해주세요.",
        }
        st.info(descriptions.get(pattern, ""))

    # ── 회복 곡선 + 상세 분석 ─────────────────
    if len(records_sorted) >= 2:
        st.divider()
        st.subheader("📈 회복 곡선")
        fig = make_recovery_curve(records_sorted, stable_day, stable_method, pattern)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("📊 상세 회복 곡선 분석", expanded=False):
            _a0 = areas[0] if areas[0] > 0 else 1
            _total_reduction = (_a0 - areas[-1]) / _a0 * 100
            _duration = days[-1] - days[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("총 관찰 기간", f"{_duration}일")
            c2.metric("누적 면적 감소율", f"{_total_reduction:.1f}%")
            c3.metric("RVI 점수", f"{rvi:.1f} / 100")

            st.write("**구간별 변화율**")
            for i in range(1, len(areas)):
                _prev = areas[i - 1]
                _chg = (areas[i] - _prev) / _prev * 100 if _prev > 0 else 0
                _icon = "🔻" if _chg < 0 else ("🔺" if _chg > 0 else "➡️")
                st.write(
                    f"Day {days[i - 1]} → Day {days[i]}: "
                    f"{_icon} **{_chg:+.1f}%** ({areas[i]:,} px)"
                )

            _interp = {
                "normal":  "상처 면적이 지속적으로 감소하고 있습니다. 현재 치유 속도를 유지하면 예상 안정화일 내에 회복이 가능합니다.",
                "plateau": "최근 상처 크기 변화가 미미합니다. 치료 계획 재검토를 권장합니다.",
                "rebound": "중간에 상처 면적이 일시적으로 증가하였습니다. 악화 원인 파악 및 추가 관찰이 필요합니다.",
                "unknown": "데이터가 부족합니다. 더 많은 시점의 이미지를 업로드해주세요.",
            }.get(pattern, "")
            if _interp:
                st.info(_interp)

    # ── 인스턴스별 분리 분석 ─────────────────
    max_inst_count = max((r.get("num_instances", 1) for r in records_sorted), default=1)
    if len(records_sorted) >= 2 and max_inst_count >= 2:
        st.divider()
        st.subheader("🔬 인스턴스별 분리 분석")
        inst_series = get_instance_series(records_sorted)
        tabs = st.tabs([f"상처 #{i + 1}" for i in range(len(inst_series))])
        for tab, inst, color in zip(tabs, inst_series, INSTANCE_COLORS_RGB):
            with tab:
                inst_days  = inst["days"]
                inst_areas = inst["areas"]
                inst_pattern = classify_pattern(inst_areas)
                inst_rvi     = compute_rvi(inst_areas, inst_days)
                if len(records_sorted) >= 2:
                    inst_stable_day, inst_stable_method = estimate_stable_day(inst_areas, inst_days)
                else:
                    inst_stable_day, inst_stable_method = inst_days[-1], "unknown"

                c1, c2, c3 = st.columns(3)
                c1.metric("패턴", PATTERN_KR.get(inst_pattern, "?"))
                c2.metric("RVI", f"{inst_rvi:.1f} / 100")
                if inst_stable_method != "unknown":
                    c3.metric("예상 안정화일", f"Day {inst_stable_day}")

                inst_records = [{"day": d, "area": a} for d, a in zip(inst_days, inst_areas)]
                fig = make_recovery_curve(inst_records, inst_stable_day, inst_stable_method, inst_pattern)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # ── 업로드된 전체 이미지 히스토리 ─────────
    if len(records_sorted) > 1:
        st.divider()
        st.subheader("📁 업로드된 이미지 히스토리")
        cols = st.columns(min(len(records_sorted), 4))
        for col, r in zip(cols, records_sorted):
            with col:
                st.image(r["overlay"], caption=f"Day {r['day']} | {r['area']:,} px",
                         use_container_width=True)


if __name__ == "__main__":
    main()
