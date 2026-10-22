"""Ajuste de compatibilidad diffusers <-> torchao (sin diffusers ni torchao reales: módulos simulados)."""
import sys
import types

import pytest

from p2 import env_check
from p2.compat import neutralize_incompatible_torchao


def fake_modules(monkeypatch, torchao_installed=True, has_fqn=False):
    iu = types.ModuleType("diffusers.utils.import_utils")
    iu._torchao_available, iu._torchao_version = torchao_installed, "0.10.0"
    iu.is_torchao_available = lambda: iu._torchao_available  # como diffusers: lee el global en cada llamada
    diffusers, utils = types.ModuleType("diffusers"), types.ModuleType("diffusers.utils")
    utils.import_utils, diffusers.utils = iu, utils
    q = types.ModuleType("torchao.quantization")
    if has_fqn:
        q.FqnToConfig = object
    torchao = types.ModuleType("torchao")
    torchao.quantization = q
    for name, mod in {"diffusers": diffusers, "diffusers.utils": utils, "diffusers.utils.import_utils": iu,
                      "torchao": torchao, "torchao.quantization": q}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return iu


def test_old_torchao_is_neutralized(monkeypatch):
    iu = fake_modules(monkeypatch, has_fqn=False)
    msg = neutralize_incompatible_torchao()
    assert iu.is_torchao_available() is False and "no tiene FqnToConfig" in msg


def test_compatible_torchao_untouched(monkeypatch):
    iu = fake_modules(monkeypatch, has_fqn=True)
    assert "sin cambios" in neutralize_incompatible_torchao() and iu.is_torchao_available() is True


def test_no_torchao_untouched(monkeypatch):
    iu = fake_modules(monkeypatch, torchao_installed=False)
    assert "no instalado" in neutralize_incompatible_torchao() and iu.is_torchao_available() is False


def test_import_smoke_failure_blocks_validation(monkeypatch):
    # En este Python no hay diffusers real: la prueba de importación debe fallar y bloquear.
    pytest.importorskip("yaml")
    for m in [k for k in sys.modules if k == "diffusers" or k.startswith("diffusers.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setitem(sys.modules, "diffusers", None)  # import diffusers -> ImportError
    state = env_check.import_smoke({"ok": True, "problems": []})
    assert state["ok"] is False and state["import_smoke"]["ok"] is False
    assert state["problems"][0].startswith("Importación fallida")
