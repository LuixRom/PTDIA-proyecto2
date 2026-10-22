"""Inspecciona el ZIP que realmente se genera (no las fuentes) para evitar entregar lógica antigua a Colab."""
import json
import re
import zipfile

import pytest
import yaml

from conftest import ROOT
from test_env_check import COLAB, fake
from p2 import __version__
from p2.env_check import validate
from tools import make_zip


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = make_zip.main(tmp_path_factory.mktemp("dist") / "p.zip")
    return zipfile.ZipFile(out)


def _cfg_from_zip(z):
    text = z.read("config/pilot.yaml").decode("utf-8")
    cfg = yaml.safe_load(text)
    cfg["_sha256"] = "zip"
    return cfg


def test_zip_has_range_not_exact_pin(built):
    cfg = _cfg_from_zip(built)
    assert cfg["dependencies"]["required"]["diffusers"] == ">=0.40.0,<0.42"
    assert not re.search(r'^\s*diffusers:\s*"0\.40\.0"\s*$', built.read("config/pilot.yaml").decode(), re.M)


def test_zip_has_no_old_exact_logic(built):
    for name in built.namelist():
        if name.endswith((".py", ".ipynb", ".yaml")) and not name.startswith("tests/"):
            text = built.read(name).decode("utf-8")
            assert not re.search(r"\bcompat_plan\s*\(|--dry-run", text), name  # ni definición ni llamada
    assert "def validate(" in built.read("p2/env_check.py").decode()


def test_zip_notebook_uses_new_validation_and_matching_version(built):
    nb = json.loads(built.read("run_colab.ipynb"))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    cell4 = next(c for c in cells if c.startswith("#@title 4."))
    assert "env_check.validate(cfg)" in cell4
    zip_init = built.read("p2/__init__.py").decode()
    zip_version = re.search(r'^__version__ = "([^"]+)"', zip_init, re.M).group(1)
    assert zip_version == __version__
    assert f'EXPECTED_P2_VERSION = "{zip_version}"' in cells[1]
    cell2 = next(c for c in cells if c.startswith("#@title 2."))
    assert "del sys.modules[mod]" in cell2 and "EXPECTED_P2_VERSION" in cell2
    assert cell2.index("del sys.modules[mod]") < cell2.index("import p2") < cell2.index("Código verificado")


@pytest.mark.parametrize("ver,ok", [("0.40.0", True), ("0.41.0", True), ("0.41.5", True),
                                    ("0.39.9", False), ("0.42.0", False), ("1.0.0", False)])
def test_zip_config_accepts_041_and_blocks_out_of_range(built, ver, ok):
    s = validate(_cfg_from_zip(built), *fake({**COLAB, "diffusers": ver}))
    assert s["ok"] is ok, (s["problems"], s["uncertain"])
    if not ok:
        assert any(p.startswith(f"diffusers {ver} fuera del rango") for p in s["problems"])


def test_zip_manifest_is_consistent(built):
    import hashlib
    for line in built.read("MANIFEST.sha256").decode().splitlines():
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256(built.read(rel)).hexdigest() == digest, rel


def test_make_zip_refuses_stale_notebook(tmp_path):
    assert make_zip.notebook_is_current() is True
    stale = tmp_path / "nb.ipynb"
    stale.write_bytes((ROOT / "run_colab.ipynb").read_bytes().replace(b"env_check.validate(cfg)", b"env_check.compat_plan()"))
    assert make_zip.notebook_is_current(stale) is False


def test_build_notebook_refuses_unregistered_data(monkeypatch, tmp_path):
    """Una foto sin registrar en data/sfhq_t2i nunca debe acabar incrustada en el notebook."""
    from tools import build_notebook
    monkeypatch.setattr(make_zip, "data_problems", lambda *a, **k: ["imagen no registrada: foto_de_internet.jpg"])
    with pytest.raises(SystemExit, match="foto_de_internet.jpg"):
        build_notebook.build(tmp_path / "nb.ipynb")
    assert not (tmp_path / "nb.ipynb").exists()
