"""Validación y preprocesamiento de la imagen de entrada."""
import hashlib
import io

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE = 256


class InvalidImage(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pad_to(img, ratio):
    """Completa la imagen hasta la proporción ancho/alto `ratio` con el color del borde (mediana), sin recortar:
    la persona queda entera. Recortar al centro una foto vertical de cuerpo entero le cortaría la cabeza."""
    import numpy as np
    w, h = img.size
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    color = tuple(int(c) for c in np.median(border, axis=0))
    W, H = (w, round(w / ratio)) if w / h > ratio else (round(h * ratio), h)
    canvas = Image.new("RGB", (W, H), color)
    canvas.paste(img, ((W - w) // 2, (H - h) // 2))
    return canvas


def prepare(data: bytes, size=512, fit="crop"):
    """bytes -> (imagen RGB sin EXIF, sha256 de los bytes originales).
    fit="crop": recorte cuadrado central size×size (retratos, versión 0).
    fit="pad" (probador): sin recortar; fotos verticales -> size×1.5·size (512×768, formato retrato que SD 1.5
    genera bien: la persona ocupa más píxeles), el resto -> size×size."""
    if len(data) > MAX_BYTES:
        raise InvalidImage(f"La imagen pesa {len(data) / 1e6:.1f} MB; el máximo es {MAX_BYTES / 1e6:.0f} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImage("El archivo no es una imagen legible.") from e
    if img.format not in ALLOWED_FORMATS:
        raise InvalidImage(f"Formato {img.format} no admitido; use {sorted(ALLOWED_FORMATS)}.")
    if min(img.size) < MIN_SIDE:
        raise InvalidImage(f"La imagen mide {img.size}; el lado mínimo es {MIN_SIDE}px.")
    img = img.convert("RGB")
    w, h = img.size
    if fit == "pad":
        portrait = h > 1.2 * w
        target = (size, size * 3 // 2) if portrait else (size, size)
        img = pad_to(img, target[0] / target[1]).resize(target, Image.Resampling.LANCZOS)
    else:
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    clean = Image.new("RGB", img.size)  # imagen nueva: no arrastra EXIF ni metadatos
    clean.paste(img)
    return clean, sha256_bytes(data)


def count_faces(img):
    """Número de caras frontales (OpenCV Haar). None si no está disponible: falta cv2, o es OpenCV 5,
    donde CascadeClassifier ya no viene en el paquete principal. Es una métrica orientativa."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return None
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return len(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)))
