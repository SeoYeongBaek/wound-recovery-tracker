"""Mask R-CNN 구성 / 가중치 로드 / 인스턴스 추론."""

from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from wound.config import MASK_THR, MAX_INSTANCES, NUM_CLASSES, SCORE_THR
from wound.postprocess import postprocess_mask


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True,
                freeze_backbone: bool = True, verbose: bool = True):
    """
    Mask R-CNN ResNet-50 FPN 모델을 구성.

    Args:
        num_classes (int): 배경 포함 클래스 수 (wound=2)
        pretrained (bool): COCO 사전학습 가중치 사용 여부
        freeze_backbone (bool): backbone(ResNet-50 + FPN) 파라미터 고정 여부
        verbose (bool): 동결/학습 파라미터 수 출력 여부
    Returns:
        torch.nn.Module: 구성된 Mask R-CNN 모델
    """
    # pretrained=False(가중치 로드용)일 때 ImageNet 백본도 다운로드하지 않는다
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
        if verbose:
            frozen = sum(1 for n, p in model.named_parameters()
                         if "backbone" in n and not p.requires_grad)
            print(f"Backbone 동결 파라미터 수: {frozen}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"학습 파라미터: {trainable:,} / 전체: {total:,}")
    return model


def load_model(model_path: str, device: torch.device = None):
    """
    학습된 state_dict를 읽어 추론용(eval) 모델을 반환.

    Returns:
        (model, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(NUM_CLASSES, pretrained=False, freeze_backbone=False, verbose=False)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def predict_instances(model, device, rgb: np.ndarray,
                      score_thr: float = SCORE_THR, mask_thr: float = MASK_THR,
                      max_instances: int = MAX_INSTANCES,
                      postprocess: bool = True) -> Tuple[List[np.ndarray], List[float]]:
    """
    RGB 이미지 한 장에 추론을 실행하고 인스턴스별 binary 마스크를 반환.

    Args:
        rgb (np.ndarray): H×W×3 uint8 RGB 이미지 (리사이즈는 호출자가 수행)
        score_thr / mask_thr: 검출 score / 마스크 확률 임계값
        max_instances: score 상위 최대 인스턴스 수
        postprocess: postprocess_mask 적용 여부
    Returns:
        (masks, scores): score 내림차순. 후처리 후 빈 마스크는 제외.
    """
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)

    with torch.inference_mode():
        out = model([img_t])[0]

    scores_np  = out.get("scores", torch.tensor([])).detach().cpu().numpy()
    masks_tens = out.get("masks", None)

    masks, scores = [], []
    if len(scores_np) > 0 and masks_tens is not None:
        keep_idx = np.where(scores_np >= score_thr)[0][:max_instances]
        for ki in keep_idx:
            m = (masks_tens[ki, 0].detach().cpu().numpy() > mask_thr).astype(np.uint8)
            if postprocess:
                m = postprocess_mask(m)
            if m.sum() > 0:
                masks.append(m)
                scores.append(float(scores_np[ki]))
    return masks, scores
