import json
import re

from conftest import ROOT
from tools.build_notebook import build

NB = ROOT / "run_colab.ipynb"


def _cells():
    return json.loads(NB.read_text(encoding="utf-8"))["cells"]


def _code(cells):
    return [("".join(c["source"])) for c in cells if c["cell_type"] == "code"]


def test_notebook_is_up_to_date(tmp_path):
    fresh = build(tmp_path / "nb.ipynb")
    assert fresh.read_text(encoding="utf-8") == NB.read_text(encoding="utf-8"), "Ejecute tools/build_notebook.py"


def test_notebook_safe_defaults_and_no_local_paths():
    src = "\n".join("".join(c["source"]) for c in _cells())
    # Descarga y ejecución del piloto: siempre desactivadas por defecto.
    for flag in ("AUTORIZO_DESCARGAR_MODELOS", "AUTORIZO_EJECUTAR_PILOTO", "AUTORIZO_APP", "EXPORTAR_A_DRIVE"):
        assert re.search(rf"^{flag} = False", src, re.M), flag
    # Continuar sin LPIPS: decisión del usuario (lpips no se instala); solo omite esa métrica.
    assert re.search(r"^ACEPTO_SIN_LPIPS = True", src, re.M)
    assert re.search(r'^FUENTE_CODIGO = "incluido"', src, re.M)
    assert not re.search(r"[A-Za-z]:\\\\|\\Users\\|C:/", src), "ruta de Windows en el notebook"
    assert not re.search(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|api[_-]?key\s*=)", src, re.I), "credencial incrustada"
    assert all(c.get("outputs", []) == [] and c.get("execution_count") is None for c in _cells() if c["cell_type"] == "code")


def test_notebook_never_runs_pip_or_dry_run():
    code = "\n".join(_code(_cells()))
    assert not re.search(r"-m\s*[\"',]*\s*pip|[!%]pip|\bpip\s+install|--dry-run|subprocess|os\.system", code), \
        "el notebook no debe invocar pip ni procesos externos"
    reqs = [l.split("#")[0].strip() for l in (ROOT / "requirements" / "colab.txt").read_text(encoding="utf-8").splitlines()]
    assert [r for r in reqs if r] == []


def test_cells_6_and_7_require_validation_before_models():
    code = _code(_cells())
    gen = next(c for c in code if "generate_all" in c)
    met = next(c for c in code if "metrics.Models" in c)
    for c, first_model_use in ((gen, "generate_all("), (met, "metrics.Models(")):
        assert "require_validation" in c and c.index("require_validation") < c.index(first_model_use)
    assert "validation=validation" in gen and "validation=validation" in met
    assert "ACEPTO_SIN_LPIPS" in met and met.index("ACEPTO_SIN_LPIPS") < met.index("metrics.Models(")
    val = next(c for c in code if "env_check.validate(" in c)
    assert "unlink(missing_ok=True)" in val and "SystemExit" in val


def test_config_invariants(cfg):
    assert cfg["num_variations"] == 5
    for m in cfg["methods"].values():
        assert len(m["variations"]) == 5
    for k in ("sd15", "clip"):
        assert re.fullmatch(r"[0-9a-f]{40}", cfg["models"][k]["revision"])
    g = cfg["generation"]
    assert g["safety_checker"] is True and g["scheduler"] == "euler"
    assert (g["steps"], g["strength"], g["guidance_scale"]) == (20, 0.40, 7.0)
    assert cfg["dataset"]["allowed_models"] == ["SDXL"]
    assert len(cfg["attribute_texts"]) == 5
    d = cfg["dependencies"]
    assert d["required"]["diffusers"] == ">=0.40.0,<0.42" and d["optional"] == {"lpips": "==0.1.4"}
    assert {"diffusers", "transformers", "accelerate"} <= set(d["verify_declared_requirements_of"])


def test_notebook_embeds_current_code_payload():
    """El código incluido (celda 1b) es exactamente el ZIP del proyecto actual, y la celda 2 lo verifica."""
    import base64
    import hashlib
    import io
    import zipfile
    from tools.make_zip import payload_bytes
    cells = _cells()
    c1b = next("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("#@title 1b."))
    b64 = re.search(r'^PAYLOAD_B64 = "([A-Za-z0-9+/=]+)"', c1b, re.M).group(1)
    sha = re.search(r'^PAYLOAD_SHA256 = "([0-9a-f]{64})"', c1b, re.M).group(1)
    data = base64.b64decode(b64)
    assert hashlib.sha256(data).hexdigest() == sha and data == payload_bytes()
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert {"p2/__init__.py", "config/pilot.yaml", "data/sfhq_t2i/SDXL_image_0035846.jpg", "MANIFEST.sha256"} <= names
    assert "run_colab.ipynb" not in names
    cell2 = next("".join(c["source"]) for c in cells if "".join(c["source"]).startswith("#@title 2."))
    assert "PAYLOAD_SHA256" in cell2 and "os.chdir(Path(WORKDIR).parent)" in cell2


def test_every_later_cell_has_order_guard():
    for c in _cells():
        s = "".join(c["source"])
        if c["cell_type"] == "code" and not s.startswith(("#@title 1.", "#@title 1b.")):
            assert 'if "requiere" not in globals()' in s.splitlines()[1], s.splitlines()[0]
            assert c["metadata"].get("cellView") == "form"


def test_notebook_has_no_pandas_or_ipython_dependency():
    code = "\n".join(_code(_cells()))
    assert "import pandas" not in code and "pandas as pd" not in code and "IPython" not in code


def test_unauthorized_pilot_cells_skip_without_exceptions():
    """Sin autorización, 6-8 no lanzan SystemExit (Colab lo muestra como error): solo avisan y se omiten."""
    for c in _cells():
        s = "".join(c["source"])
        if s.startswith(("#@title 6.", "#@title 7.", "#@title 8.")):
            lines = s.splitlines()
            gate = lines.index("if not PILOTO_AUTORIZADO:")
            assert lines[gate + 1].startswith("    print(") and lines[gate + 2] == "else:"
            assert "raise" not in "\n".join(lines[gate:gate + 2])
    assert "no autorizado (es lo esperado" not in "\n".join(_code(_cells()))
    cell1 = next("".join(c["source"]) for c in _cells() if "".join(c["source"]).startswith("#@title 1."))
    assert "PILOTO_AUTORIZADO = bool(AUTORIZO_DESCARGAR_MODELOS and AUTORIZO_EJECUTAR_PILOTO)" in cell1


def test_app_cells_gated_and_inline():
    cells = {"".join(c["source"]).split("\n")[0]: "".join(c["source"]) for c in _cells()}
    c1 = next(v for k, v in cells.items() if k.startswith("#@title 1. "))
    assert re.search(r"^AUTORIZO_APP = False", c1, re.M)
    assert "APP_AUTORIZADA = bool(AUTORIZO_DESCARGAR_MODELOS and AUTORIZO_APP)" in c1
    for prefix in ("#@title 9. ", "#@title 10. "):
        s = next(v for k, v in cells.items() if k.startswith(prefix))
        lines = s.splitlines()
        g = lines.index("if not APP_AUTORIZADA:")
        assert lines[g + 1].startswith("    print(") and lines[g + 2] == "else:"
    app = next(v for k, v in cells.items() if k.startswith("#@title 9. "))
    assert "share=False" in app and "inline=True" in app and "require_validation" in app
    # Regresión (Colab): sin allowed_paths gradio rechaza servir las imágenes de /content/sesiones (InvalidPathError).
    assert "allowed_paths=[str(store.root)]" in app and "show_error=True" in app
    assert 'cfg["app"]["strength"]' in app and "load_llm(cfg)" in app


def test_app_and_llm_config(cfg):
    assert cfg["app"] == {"method": "A2", "strength": 0.55, "sessions_dir": "sesiones"}
    assert cfg["llm"]["repo"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert re.fullmatch(r"[0-9a-f]{40}", cfg["llm"]["revision"]) and cfg["llm"]["license"] == "Apache-2.0"
