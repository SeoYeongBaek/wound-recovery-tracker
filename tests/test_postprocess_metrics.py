import numpy as np
import pytest

from wound.metrics import compute_metrics
from wound.postprocess import postprocess_mask


def test_postprocess_removes_small_components_and_keeps_large():
    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:40, 10:40] = 1    # 900 px → 유지
    m[70:75, 70:75] = 1    # 25 px  → 제거
    out = postprocess_mask(m, min_area=200)
    assert out.dtype == np.uint8
    assert out[25, 25] == 1
    assert out[72, 72] == 0


def test_postprocess_fills_small_holes():
    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:80, 20:80] = 1
    m[50, 50] = 0
    assert postprocess_mask(m)[50, 50] == 1


def test_metrics_values():
    pred = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    gt   = np.array([[0, 1, 1, 0]], dtype=np.uint8)
    iou, dice, prec, rec, tp, fp, fn = compute_metrics(pred, gt)
    assert (tp, fp, fn) == (1, 1, 1)
    assert iou == pytest.approx(1 / 3)
    assert dice == pytest.approx(0.5)
    assert prec == pytest.approx(0.5)
    assert rec == pytest.approx(0.5)


def test_metrics_empty_masks_return_none():
    z = np.zeros((4, 4), dtype=np.uint8)
    iou, dice, prec, rec, *_ = compute_metrics(z, z)
    assert iou is None and dice is None and prec is None and rec is None
