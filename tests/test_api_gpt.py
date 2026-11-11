import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import probar_api_gpt as api  # noqa: E402


def test_key_is_read_from_env_file_with_either_name(tmp_path, monkeypatch):
    for n in api.KEY_NAMES:
        monkeypatch.delenv(n, raising=False)
    f = tmp_path / ".env"
    f.write_text('# comentario\nAPI-KEY="sk-prueba-123"\n', encoding="utf-8")
    assert api.cargar_clave(f) == "sk-prueba-123"
    f.write_text("OPENAI_API_KEY=otra\n", encoding="utf-8")
    assert api.cargar_clave(f) == "otra"
    f.write_text("API-KEY=\n", encoding="utf-8")
    assert api.cargar_clave(f) is None


def test_keys_never_appear_in_messages():
    assert "sk-proj-abcDEF123" not in api.ocultar("Incorrect API key provided: sk-proj-abcDEF123. Find it at ...")


def test_multipart_has_fields_and_file():
    cuerpo, ctype = api.multipart({"model": "m", "prompt": "p"}, {"image": ("f.png", b"\x89PNG", "image/png")})
    assert ctype.startswith("multipart/form-data; boundary=")
    b = ctype.split("boundary=")[1].encode()
    assert cuerpo.count(b"--" + b) == 4 and b'name="model"' in cuerpo and b'filename="f.png"' in cuerpo


def test_example_env_has_no_real_key():
    import re
    t = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9_\-]{20,}", t)  # un marcador corto está bien; una clave real (largo) no
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8")
