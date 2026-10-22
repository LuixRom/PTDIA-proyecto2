"""Reglas del conjunto de imágenes de entrada (data/sfhq_t2i): nombre, formato, manifiesto.

Lo usan tools/register_image.py (registrar) y tools/make_zip.py (rechazar imágenes no registradas).
No descarga nada.
"""
import csv
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .export import sha256_file

MANIFEST_FIELDS = ["image_filename", "sha256", "model_used", "style_preset", "role", "selection_reason", "source", "license"]
NAME_RE = re.compile(r"^(?P<model>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?)_image_\d{7}\.jpg$")
EXPECTED_SIZE = (1024, 1024)   # todas las imágenes de SFHQ-T2I son 1024x1024 (ficha de Kaggle)
EXPECTED_FORMAT = "JPEG"
SDXL_ENGINE = "stable-diffusion-xl-1024-v1-0"   # create_face_dataset.py del autor
PHOTO_PRESETS = {"photographic", "analog-film", "cinematic"}
NON_IMAGE_FILES = {"MANIFEST.csv", "SOURCE.md", "LICENSE_MIT.txt"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def model_from_name(name):
    m = NAME_RE.match(name)
    return m.group("model") if m else None


def image_problems(path):
    """Problemas de formato/dimensiones de una imagen de SFHQ-T2I ([] = válida)."""
    try:
        with Image.open(path) as im:
            fmt, size = im.format, im.size
            im.verify()
    except (UnidentifiedImageError, OSError) as e:
        return [f"{Path(path).name}: no es una imagen legible ({e})."]
    p = []
    if fmt != EXPECTED_FORMAT:
        p.append(f"{Path(path).name}: formato {fmt}, se esperaba {EXPECTED_FORMAT}.")
    if size != EXPECTED_SIZE:
        p.append(f"{Path(path).name}: tamaño {size}, se esperaba {EXPECTED_SIZE}.")
    return p


def read_manifest(path):
    """(filas, problemas). Exige la cabecera exacta y nombres no repetidos."""
    path = Path(path)
    if not path.exists():
        return [], [f"No existe {path}."]
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != MANIFEST_FIELDS:
            return [], [f"Cabecera de {path.name} distinta de la esperada: {MANIFEST_FIELDS}."]
        rows = list(reader)
    names = [r["image_filename"] for r in rows]
    dups = sorted({n for n in names if names.count(n) > 1})
    return rows, ([f"Nombres repetidos en {path.name}: {dups}."] if dups else [])


def check_dataset_dir(data_dir, allowed_models, pilot_image=None):
    """Coherencia carpeta <-> manifiesto. Devuelve lista de problemas ([] = todo registrado y coherente)."""
    data_dir = Path(data_dir)
    rows, probs = read_manifest(data_dir / "MANIFEST.csv")
    registered = {r["image_filename"]: r for r in rows}
    for p in sorted(x for x in data_dir.iterdir() if x.is_file()):
        if p.name in NON_IMAGE_FILES:
            continue
        if p.name not in registered:
            kind = "imagen" if p.suffix.lower() in IMAGE_EXT else "archivo"
            probs.append(f"{kind} no registrada en MANIFEST.csv: {p.name} (use tools/register_image.py o retírela).")
    for name, r in registered.items():
        p = data_dir / name
        if not p.exists():
            probs.append(f"MANIFEST.csv lista {name}, pero el archivo no existe.")
            continue
        if sha256_file(p) != r["sha256"]:
            probs.append(f"SHA-256 de {name} no coincide con MANIFEST.csv.")
        if r["model_used"] not in allowed_models or model_from_name(name) != r["model_used"]:
            probs.append(f"{name}: model_used '{r['model_used']}' no coincide con el nombre o no está permitido {allowed_models}.")
        probs += image_problems(p)
    pilots = [n for n, r in registered.items() if r["role"] == "pilot"]
    if len(pilots) > 1:
        probs.append(f"Hay más de una imagen con role=pilot: {pilots}.")
    if pilot_image and pilot_image not in registered:
        probs.append(f"dataset.pilot_image = {pilot_image} no está registrada en MANIFEST.csv.")
    return probs
