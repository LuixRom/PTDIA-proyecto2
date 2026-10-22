"""Empaqueta el proyecto para Colab en dist/proyecto2_colab.zip (con MANIFEST.sha256). Uso: python tools/make_zip.py

Solo incluye código, configuración, tests, documentación de uso y data/sfhq_t2i/.
No incluye Untitled1.ipynb, Material/, doc-principal/ ni planificacion/.
"""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from p2.export import write_manifest, zip_dir  # noqa: E402

INCLUDE = ["p2", "experiments", "tools", "config", "requirements", "data/sfhq_t2i", "tests",
           "README.md", "run_colab.ipynb", "pytest.ini"]
SKIP = {"__pycache__", ".pytest_cache"}


def stage(dst, include=INCLUDE):
    for rel in include:
        src = ROOT / rel
        if src.is_dir():
            shutil.copytree(src, dst / rel, ignore=shutil.ignore_patterns(*SKIP, "*.pyc"))
        elif src.exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst / rel)
        else:
            raise FileNotFoundError(f"Falta {rel}; genere run_colab.ipynb con tools/build_notebook.py")
    return write_manifest(dst)


def payload_bytes():
    """ZIP del proyecto SIN run_colab.ipynb: es lo que el notebook lleva incluido (celda 1b)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "payload.zip"
        stage(src, [i for i in INCLUDE if i != "run_colab.ipynb"])
        zip_dir(src, out)
        return out.read_bytes()


def notebook_is_current(nb_path=ROOT / "run_colab.ipynb"):
    """True si el notebook en disco es exactamente el que genera tools/build_notebook.py ahora."""
    from tools.build_notebook import build
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build(Path(tmp) / "nb.ipynb")
        return Path(nb_path).exists() and fresh.read_bytes() == Path(nb_path).read_bytes()


def data_problems(config_path=ROOT / "config" / "pilot.yaml"):
    """Todo archivo de data/sfhq_t2i debe estar registrado y coherente (p2.dataset). No descarga nada."""
    import yaml
    from p2.dataset import check_dataset_dir
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    d = cfg["dataset"]
    return check_dataset_dir(Path(config_path).parent.parent / d["dir"], d["allowed_models"], d["pilot_image"])


def main(out=ROOT / "dist" / "proyecto2_colab.zip"):
    if not notebook_is_current():
        raise SystemExit("run_colab.ipynb no coincide con tools/build_notebook.py: ejecute "
                         "`python tools/build_notebook.py` antes de empaquetar.")
    probs = data_problems()
    if probs:
        raise SystemExit("ZIP no generado; data/sfhq_t2i no es coherente:\n- " + "\n- ".join(probs))
    with tempfile.TemporaryDirectory() as tmp:
        n = stage(Path(tmp))
        zip_dir(tmp, out)
    print(f"{out} ({out.stat().st_size / 1e3:.1f} kB, {n} archivos + MANIFEST.sha256)")
    return out


if __name__ == "__main__":
    main()
