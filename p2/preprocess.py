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


def prepare(data: bytes, size=512):
    """bytes -> (imagen RGB size×size sin EXIF, sha256 de los bytes originales)."""
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
