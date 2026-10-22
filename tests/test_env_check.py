import json
import re
import socket

import pytest

from p2 import env_check
from p2.env_check import (ValidationError, check_declared_requirements, collect, problems, require_validation,
                          save_validation, validate)

REQ = {"min_vram_gb": 12, "min_ram_gb": 10, "min_disk_gb": 10}


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


# Versiones observadas por el usuario en Colab (torch, diffusers, huggingface-hub, lpips ausente).
# El resto no se informó: se asumen valores típicos preinstalados (supuesto de test).
COLAB = {"torch": "2.11.0+cu130", "diffusers": "0.41.0", "huggingface-hub": "1.33.0", "lpips": None,
         "torchvision": "0.26.0+cu130", "transformers": "5.20.2", "accelerate": "1.15.0", "safetensors": "0.8.1",
         "numpy": "2.2.0", "pillow": "11.3.0", "scikit-image": "0.25.2", "opencv-python": "4.12.0.88",
         "scipy": "1.16.0", "psutil": "7.0.0", "pydantic": "2.12.0", "pyyaml": "6.0.2", "packaging": "25.0",
         "tokenizers": "0.23.1", "tqdm": "4.67.1", "importlib-metadata": "8.7.0", "filelock": "3.19.1",
         "regex": "2025.11.3", "requests": "2.32.4", "typer": "0.16.0"}

# Dependencias declaradas (PyPI, verificado 2026-10-21) + un extra que debe ignorarse.
REQUIRES = {
    "diffusers": ["importlib_metadata", "filelock", "huggingface-hub<2.0,>=1.32.0", "numpy", "regex!=2019.12.17",
                  "requests", "safetensors>=0.8.0", "Pillow", 'torch>=1.4; extra == "torch"'],
    "transformers": ["huggingface-hub<3.0,>=1.31.0", "numpy>=1.17", "packaging>=20.0", "pyyaml>=5.1",
                     "regex>=2025.10.22", "tokenizers<0.24.0,>=0.23.1", "typer", "safetensors>=0.8.0", "tqdm>=4.60"],
    "accelerate": ["numpy>=1.17", "packaging>=20.0", "psutil", "pyyaml", "torch>=2.0.0", "huggingface_hub>=0.21.0",
                   "safetensors>=0.4.3"],
    "lpips": ["torch (>=0.4.0)", "torchvision (>=0.2.1)", "numpy (>=1.14.3)", "scipy (>=1.0.1)", "tqdm (>=4.28.1)"],
}


def fake(versions, requires=REQUIRES):
    vs = {norm(k): v for k, v in versions.items()}

    def version_fn(n):
        return vs.get(norm(n))

    def requires_fn(n):
        if vs.get(norm(n)) is None:
            raise env_check.metadata.PackageNotFoundError(n)
        r = requires.get(norm(n), [])
        if isinstance(r, Exception):
            raise r
        return r
    return version_fn, requires_fn


# --- recursos ---------------------------------------------------------------

def info(**kw):
    base = {"cuda": True, "gpu": "Tesla T4", "vram_total_gb": 15.6, "ram_total_gb": 13.6, "disk_free_gb": 70,
            "packages": {"torch": "2.11.0"}}
    base.update(kw)
    return base


def test_t4_ok():
    assert problems(info(), REQ) == []


@pytest.mark.parametrize("kw,needle", [
    ({"cuda": False}, "No hay GPU"), ({"vram_total_gb": 4.0}, "VRAM"), ({"ram_total_gb": 8.0}, "RAM"),
    ({"disk_free_gb": 3.0}, "Disco"), ({"packages": {"torch": None}}, "PyTorch no está instalado"),
])
def test_explanatory_errors(kw, needle):
    assert any(needle in p for p in problems(info(**kw), REQ))


def test_collect_runs_without_torch():
    i = collect(".")
    assert i["disk_free_gb"] > 0 and "packages" in i


# --- validación de compatibilidad --------------------------------------------

def test_user_colab_versions_validate_ok(cfg):
    s = validate(cfg, *fake(COLAB))
    assert s["ok"], (s["problems"], s["uncertain"])
    assert s["versions"]["diffusers"] == "0.41.0" and s["versions"]["huggingface-hub"] == "1.33.0"
    assert s["optional"]["lpips"]["available"] is False and s["network_used"] is False
    assert any("diffusers → huggingface-hub<2.0,>=1.32.0 (1.33.0)" in c for c in s["checked"])
    assert not any("extra" in c for c in s["checked"])  # el extra "torch" se ignora


def test_hub2_blocks_because_diffusers_declares_lt2(cfg):
    s = validate(cfg, *fake({**COLAB, "huggingface-hub": "2.2.0"}))
    assert not s["ok"] and any("diffusers requiere huggingface-hub" in p for p in s["problems"])


@pytest.mark.parametrize("ver", ["0.39.0", "0.42.0"])
def test_diffusers_outside_accepted_range(cfg, ver):
    s = validate(cfg, *fake({**COLAB, "diffusers": ver}))
    assert not s["ok"] and any(p.startswith(f"diffusers {ver} fuera del rango") for p in s["problems"])


def test_old_torch_blocks(cfg):
    s = validate(cfg, *fake({**COLAB, "torch": "2.2.1"}))
    assert not s["ok"] and any(p.startswith("torch 2.2.1") for p in s["problems"])


def test_missing_declared_dependency_blocks(cfg):
    s = validate(cfg, *fake({**COLAB, "typer": None}))
    assert not s["ok"] and any("transformers requiere typer" in p for p in s["problems"])


def test_unreadable_metadata_is_uncertain_and_blocks(cfg):
    s = validate(cfg, *fake(COLAB, {**REQUIRES, "diffusers": OSError("METADATA dañado")}))
    assert not s["ok"] and not s["problems"] and any("no se pudieron leer" in u for u in s["uncertain"])


def test_unparsable_requirement_or_version_is_uncertain(cfg):
    s = validate(cfg, *fake(COLAB, {**REQUIRES, "accelerate": ["torch>>>2"]}))
    assert not s["ok"] and any("no interpretable" in u for u in s["uncertain"])
    s = validate(cfg, *fake({**COLAB, "torch": "versión-rara"}))
    assert not s["ok"] and any("no se puede interpretar" in u for u in s["uncertain"])


def test_lpips_optional_available_when_installed(cfg):
    s = validate(cfg, *fake({**COLAB, "lpips": "0.1.4"}))
    assert s["ok"] and s["optional"]["lpips"]["available"]
    s = validate(cfg, *fake({**COLAB, "lpips": "0.1.3"}))
    assert s["ok"] and not s["optional"]["lpips"]["available"]  # opcional: no bloquea la validación


def test_check_declared_requirements_direct():
    p, u, c = check_declared_requirements("lpips", *fake({**COLAB, "lpips": "0.1.4"}))
    assert p == [] and u == [] and len(c) == 5


def test_validation_uses_no_network(cfg, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("acceso a red durante la validación")
    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert validate(cfg, *fake(COLAB))["ok"]
    s = validate(cfg)  # metadatos REALES de este equipo (sin torch): debe terminar, sin red, y no ser ok
    assert s["ok"] is False and any(p.startswith("torch no está instalado") for p in s["problems"])


# --- estado de validación que exigen las celdas 6 y 7 -------------------------

def test_require_validation_states(cfg, tmp_path):
    vf, rf = fake(COLAB)
    path = tmp_path / "validation.json"
    with pytest.raises(ValidationError, match="celda 4"):
        require_validation(cfg, path, vf)                       # nunca se validó
    save_validation(validate(cfg, vf, rf), path)
    assert require_validation(cfg, path, vf)["ok"]              # validación correcta

    bad = validate(cfg, *fake({**COLAB, "huggingface-hub": "2.2.0"}))
    save_validation(bad, path)
    with pytest.raises(ValidationError, match="no fue correcta"):
        require_validation(cfg, path, vf)

    save_validation(validate(cfg, vf, rf), path)
    with pytest.raises(ValidationError, match="configuración cambió"):
        require_validation({**cfg, "_sha256": "otro"}, path, vf)
    vf2, _ = fake({**COLAB, "lpips": "0.1.4"})                  # alguien instaló algo después de validar
    with pytest.raises(ValidationError, match="paquetes instalados cambiaron"):
        require_validation(cfg, path, vf2)


def test_saved_validation_is_json(cfg, tmp_path):
    p = save_validation(validate(cfg, *fake(COLAB)), tmp_path / "env" / "v.json")
    assert json.loads(p.read_text(encoding="utf-8"))["config_sha256"] == cfg["_sha256"]
