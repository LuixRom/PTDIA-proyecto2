import io

import pytest
from PIL import Image

from conftest import image_bytes
from p2.preprocess import InvalidImage, count_faces, prepare


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGB"), ("JPEG", "RGB"), ("PNG", "RGBA"), ("PNG", "L"), ("WEBP", "RGB")])
def test_valid_formats_to_512_rgb(fmt, mode):
    img, sha = prepare(image_bytes(fmt, mode=mode))
    assert img.size == (512, 512) and img.mode == "RGB" and len(sha) == 64


def test_exif_removed():
    src = Image.new("RGB", (600, 600), "red")
    exif = Image.Exif()
    exif[0x010F] = "CamaraFalsa"
    buf = io.BytesIO()
    src.save(buf, "JPEG", exif=exif)
    assert Image.open(io.BytesIO(buf.getvalue())).getexif()
    img, _ = prepare(buf.getvalue())
    assert not img.getexif() and not img.info.get("exif")


def test_sha_is_stable():
    b = image_bytes()
    assert prepare(b)[1] == prepare(b)[1]


@pytest.mark.parametrize("data,msg", [
    (b"no soy una imagen", "no es una imagen"),
    (image_bytes(size=(100, 300)), "lado mínimo"),
    (image_bytes("GIF"), "no admitido"),
], ids=["no_imagen", "pequena", "gif"])
def test_invalid(data, msg):
    with pytest.raises(InvalidImage, match=msg):
        prepare(data)


def test_too_big():
    with pytest.raises(InvalidImage, match="máximo"):
        prepare(b"0" * (10 * 1024 * 1024 + 1))


def test_count_faces_unavailable_on_opencv5(monkeypatch):
    """OpenCV 5 no trae CascadeClassifier en el paquete principal: la métrica queda None, no rompe."""
    import sys
    import types
    monkeypatch.setitem(sys.modules, "cv2", types.ModuleType("cv2"))
    assert count_faces(Image.new("RGB", (64, 64))) is None


def test_count_faces_on_blank_is_zero():
    pytest.importorskip("cv2")
    assert count_faces(Image.new("RGB", (512, 512), "gray")) == 0
