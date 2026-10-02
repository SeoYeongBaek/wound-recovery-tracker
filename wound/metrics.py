"""세그멘테이션 평가 지표."""

import numpy as np


def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """
    IoU / Dice / Precision / Recall 계산.

    Args:
        pred_bin (np.ndarray): 예측 binary 마스크
        gt_bin (np.ndarray): GT binary 마스크
    Returns:
        tuple: (iou, dice, precision, recall, tp, fp, fn) — 분모가 0이면 해당 지표는 None
    """
    pred = pred_bin.astype(bool)
    gt   = gt_bin.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec  = tp / (tp + fn) if (tp + fn) > 0 else None
    return iou, dice, prec, rec, tp, fp, fn
