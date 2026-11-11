import io

import pytest
from PIL import Image

from conftest import image_bytes
from p2 import api_images, config, vitrina
from p2.decisions import DecisionStore
from p2.guardrails import GuardrailError

PHOTO = image_bytes(size=(640, 640))
OPTS = dict(mode="ocasion", occasion="reunion", linea="mujer", approach="API", seed_mode="prueba",
            own_photo=True, adult=True, api_consent=True)


@pytest.fixture
def env(tmp_path):
    cfg = config.load("config/vitrina.yaml")
    eng = vitrina.Engine.mock(cfg)
    eng.api = lambda img, texto: (img.copy(), {"total_tokens": 1000})  # API simulada: sin red ni costo
    return DecisionStore(tmp_path / "sesiones"), eng


def test_api_approach_needs_key_and_explicit_consent(env):
    store, eng = env
    with pytest.raises(GuardrailError, match="aceptas que tu foto se envíe"):
        vitrina.start_session(store, eng, PHOTO, {**OPTS, "api_consent": False})
    eng.api = None
    with pytest.raises(GuardrailError, match="falta la clave"):
        vitrina.start_session(store, eng, PHOTO, OPTS)


def test_api_session_has_five_looks_usage_and_estimated_cost(env):
    store, eng = env
    sid, paths = vitrina.start_session(store, eng, PHOTO, OPTS)
    s = store.load(sid)
    assert len(paths) == 5 and all(v.metrics["api_tokens"] == 1000 for v in s.variations)
    assert all(v.metrics["face_ssim"] == 1.0 for v in s.variations)   # rostro, cabello y fondo: los de la foto
    m = vitrina.session_metrics(store, sid, eng.cfg)
    assert m["imagenes_api"] == 5 and m["costo_api_estimado_usd"] == round(5 * eng.cfg["negocio"]["usd_por_imagen_api"], 3)


def test_api_error_on_one_look_gives_a_card_and_no_crash(env):
    store, eng = env
    calls = []

    def flaky(img, texto):
        calls.append(1)
        if len(calls) == 2:
            raise api_images.ApiError("HTTP 400: contenido no permitido")
        return img.copy(), {"total_tokens": 5}
    eng.api = flaky
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    vs = store.load(sid).variations
    assert [bool(v.metrics.get("api_error")) for v in vs].count(True) == 1 and len(vs) == 5


def test_fatal_api_error_stops_with_a_clear_message(env):
    store, eng = env

    def sin_saldo(img, texto):
        raise api_images.ApiError("HTTP 429: You have no credits remaining", fatal=True)
    eng.api = sin_saldo
    with pytest.raises(GuardrailError, match="no está disponible"):
        vitrina.start_session(store, eng, PHOTO, OPTS)


def test_editor_decodes_response_and_fatal_flag(monkeypatch):
    import base64
    buf = io.BytesIO()
    Image.new("RGB", (40, 60), (10, 20, 30)).save(buf, "PNG")
    monkeypatch.setattr(api_images, "editar", lambda *a, **k: {
        "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}], "usage": {"total_tokens": 7}})
    res, usage = api_images.ApiEditor("m", "s", clave="k")(Image.new("RGB", (20, 30)), "a white shirt")
    assert res.size == (20, 30) and usage == {"total_tokens": 7}
    assert "no existe" not in api_images.prompt_look("x") and "plain top underneath" in api_images.prompt_look("x")


def test_ui_hides_api_approach_without_key():
    import gradio  # noqa: F401
    from p2 import vitrina_ui
    cfg = config.load("config/vitrina.yaml")
    eng = vitrina.Engine.mock(cfg)
    demo = vitrina_ui.build_ui(DecisionStore(__import__("pathlib").Path("x_unused")), cfg, eng,
                               vitrina_ui.LLMs(cfg, vitrina_ui.stub_llm_loader), None)
    txt = str(demo.get_config_file()["components"])
    assert cfg["approaches"]["API"]["name"] not in txt and cfg["approaches"]["B"]["name"] in txt


def test_editor_is_built_from_the_config_exactly_like_the_app_does():
    cfg = config.load("config/vitrina.yaml")
    ed = api_images.ApiEditor(**cfg["approaches"]["API"]["api"], clave="k")   # lo que hacen app_local.py y eval_local.py
    assert ed.model == cfg["approaches"]["API"]["api"]["model"] and ed.available()


def test_api_keeps_its_own_clothes_and_only_restores_face(env):
    import numpy as np
    store, eng = env
    verde = (20, 160, 40)
    eng.api = lambda img, texto: (Image.new("RGB", img.size, verde), {"total_tokens": 1})   # la «API» pinta todo de verde
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    d = store.root / sid
    out = np.asarray(Image.open(d / "V1_r1.png").convert("RGB")).astype(int)
    inp = np.asarray(Image.open(d / "input.png").convert("RGB")).astype(int)
    h, w = out.shape[:2]
    assert abs(out[int(h * .30), int(w * .50)] - inp[int(h * .30), int(w * .50)]).max() <= 3      # rostro: el original
    assert abs(out[int(h * .85), int(w * .50)] - np.array(verde)).max() <= 3                       # ropa: la de la API, sin máscara


def test_api_draws_accessories_itself_and_skips_the_slow_local_pass(env):
    store, eng = env
    textos, locales = [], []
    eng.api = lambda img, texto: (textos.append(texto), (img.copy(), {"total_tokens": 1}))[1]
    gen = eng.gen_b
    eng.gen_b = lambda img, mask, spec, control=None: (locales.append(bool(spec.get("accessory"))), gen(img, mask, spec, control))[1]
    sid, _ = vitrina.start_session(store, eng, PHOTO, {**OPTS, "occasion": "trabajo"})
    assert any("eyeglasses" in t or "necklace" in t for t in textos)      # el texto enviado a la API incluye el accesorio
    assert not any(locales)                                              # y no hubo pasada local de accesorios
    con = [v for v in store.load(sid).variations if v.metrics["accesorios_aplicados"]]
    assert con and all(not v.metrics["accesorios_omitidos"] for v in con)
