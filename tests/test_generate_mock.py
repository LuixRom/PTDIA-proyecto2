import hashlib

import pytest
from PIL import Image

from p2.generate import cinco_variaciones, mock_generate, plan


def _h(im):
    return hashlib.sha256(im.tobytes()).hexdigest()


@pytest.mark.parametrize("method", ["A1", "A2"])
def test_plan_has_five_with_paired_seeds(cfg, method):
    specs = plan(cfg, method)
    assert [s["id"] for s in specs] == ["V1", "V2", "V3", "V4", "V5"]
    assert [s["seed"] for s in specs] == [1001, 1002, 1003, 1004, 1005]


def test_a1_a2_differ_only_in_prompt(cfg):
    a1, a2 = plan(cfg, "A1"), plan(cfg, "A2")
    assert [s["seed"] for s in a1] == [s["seed"] for s in a2]
    assert len({s["prompt"] for s in a1}) == 1 and len({s["prompt"] for s in a2}) == 5


def test_mock_exactly_five_deterministic(cfg):
    img = Image.new("RGB", (512, 512), (120, 80, 60))
    r1 = cinco_variaciones(img, plan(cfg, "A2"), mock_generate)
    r2 = cinco_variaciones(img, plan(cfg, "A2"), mock_generate)
    assert len(r1) == 5 and all(im.size == (512, 512) for _, im, _ in r1)
    assert [_h(im) for _, im, _ in r1] == [_h(im) for _, im, _ in r2]


def test_rejects_wrong_count(cfg):
    with pytest.raises(ValueError, match="exactamente 5"):
        cinco_variaciones(Image.new("RGB", (8, 8)), plan(cfg, "A1")[:4], mock_generate)


def test_unknown_method(cfg):
    with pytest.raises(KeyError):
        plan(cfg, "A9")
