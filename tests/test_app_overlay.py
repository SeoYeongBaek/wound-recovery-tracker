import numpy as np
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")

import app  # noqa: E402


def test_overlay_draws_instance_labels():
    # 이전 버그: putText가 임시 복사본에 그려져 번호 레이블이 표시되지 않았음
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[30:100, 30:100] = 1

    out = app.draw_instance_overlay(rgb, [mask])

    color = np.array(app.INSTANCE_COLORS_RGB[0]) * 0.6
    region = out[mask.astype(bool)]
    white_px = np.all(region == 255, axis=1).sum()
    assert white_px > 0, "인스턴스 번호 레이블이 그려지지 않음"
    # 레이블 외 영역은 색상 블렌딩만 적용
    assert np.allclose(out[35, 35], np.round(color), atol=1)
