import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_lora_notebook_is_valid_and_current(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import build_lora_notebook as b
    nb = json.loads(b.build(tmp_path / "nb.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 8
    for c in code:  # todas las celdas son Python puro (sin comandos mágicos) y compilan
        src = "".join(c["source"])
        compile(src, "<celda>", "exec")
        assert "!pip" not in src and "%%" not in src
    full = "".join("".join(c["source"]) for c in code)
    assert "diffusers==" not in full or "DIFFUSERS_VERSION" in full
    assert f'DIFFUSERS_VERSION = "{b.DIFFUSERS}"' in full and "%(diffusers)s" not in full
    assert "token" not in full.lower() and "api_key" not in full.lower()  # sin credenciales
    out = ROOT / "lora_entrenamiento_kaggle.ipynb"
    assert out.read_text(encoding="utf-8") == (tmp_path / "nb.ipynb").read_text(encoding="utf-8"),         "Ejecute tools/build_lora_notebook.py"


def test_lora_catalog_and_curation_are_consistent():
    cat = yaml.safe_load((ROOT / "config/lora_catalogo.yaml").read_text(encoding="utf-8"))
    keep = yaml.safe_load((ROOT / "config/lora_curado_piloto.yaml").read_text(encoding="utf-8"))
    assert set(keep) == set(cat["piloto"]["prendas"]) and set(keep) <= set(cat["mujer"]["prendas"])
    for pid, idx in keep.items():
        n = cat["piloto"]["imagenes_por_prenda"] + cat["mujer"]["prendas"][pid].get("producto_n", 0)
        assert idx == sorted(set(idx)) and all(0 <= k < n for k in idx), pid
