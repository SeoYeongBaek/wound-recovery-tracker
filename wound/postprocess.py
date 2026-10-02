"""마스크 후처리."""

import cv2
import numpy as np

from wound.config import K_CLOSE, K_OPEN, MIN_AREA


def postprocess_mask(bin_mask: np.ndarray,
                     k_close: int = K_CLOSE, k_open: int = K_OPEN,
                     min_area: int = MIN_AREA) -> np.ndarray:
    """
    Morphological closing/opening + 소형 컴포넌트 제거.

    Args:
        bin_mask (np.ndarray): uint8 binary 마스크 (0/1)
        k_close (int): closing 커널 크기
        k_open (int): opening 커널 크기
        min_area (int): 이 픽셀 수 미만의 컴포넌트는 제거
    Returns:
        np.ndarray: 후처리된 uint8 binary 마스크 (0/1)
    """
    m = (bin_mask * 255).astype(np.uint8)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    m  = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    m  = cv2.morphologyEx(m, cv2.MORPH_OPEN,  ko)
    m  = (m > 0).astype(np.uint8)
    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(num, dtype=np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        m = keep[labels]
    return m
