"""Manifiestos SHA-256 y empaquetado ZIP reproducible."""
import hashlib
import zipfile
from pathlib import Path

MANIFEST = "MANIFEST.sha256"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root):
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST)
    lines = [f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in files]
    (root / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify_manifest(root):
    """Lista de problemas ([] = todo coincide)."""
    root = Path(root)
    m = root / MANIFEST
    if not m.exists():
        return [f"Falta {MANIFEST}"]
    bad = []
    for line in m.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        p = root / rel
        if not p.exists():
            bad.append(f"Falta {rel}")
        elif sha256_file(p) != digest:
            bad.append(f"SHA-256 distinto: {rel}")
    return bad


def zip_dir(root, zip_path):
    """Empaqueta `root` con orden y fechas fijas y SIN compresión: mismo contenido -> mismos bytes,
    también entre versiones de Python (la salida de DEFLATE depende de la zlib de cada intérprete)."""
    root, zip_path = Path(root), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(x for x in root.rglob("*") if x.is_file()):
            info = zipfile.ZipInfo(p.relative_to(root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16  # permisos fijos: no dependen del sistema
            z.writestr(info, p.read_bytes())
    return zip_path
