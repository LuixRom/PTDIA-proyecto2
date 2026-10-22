import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p2 import config  # noqa: E402
from p2.schemas import Session, Variation  # noqa: E402

SHA = "a" * 64


@pytest.fixture(scope="session")
def cfg():
    return config.load()


def image_bytes(fmt="PNG", size=(640, 480), mode="RGB"):
    x = np.linspace(0, 255, size[0], dtype=np.uint8)
    arr = np.stack([np.tile(x, (size[1], 1))] * 3, axis=-1)
    im = Image.fromarray(arr, "RGB").convert(mode)
    buf = io.BytesIO()
    im.save(buf, fmt)
    return buf.getvalue()


def make_session(decisions=None, sid="s-test", reasons=None):
    """decisions: dict id -> 'aceptada'/'rechazada' (o None)."""
    decisions = decisions or {}
    reasons = reasons or {}
    attrs = ["luz_calida", "luz_fria_estudio", "acuarela", "sonrisa", "bn_epoca"]
    vs = [Variation(id=f"V{k}", method="A2", attribute=attrs[k - 1], prompt=f"prompt {k}", seed=1000 + k,
                    decision=decisions.get(f"V{k}"), reason=reasons.get(f"V{k}", ""))
          for k in range(1, 6)]
    return Session(session_id=sid, input_sha256=SHA, variations=vs)


@pytest.fixture
def mixed():
    return make_session({"V1": "aceptada", "V2": "rechazada", "V3": "rechazada", "V4": "aceptada", "V5": "rechazada"})
