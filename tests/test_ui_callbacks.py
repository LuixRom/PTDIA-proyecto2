import json

import pytest

from conftest import image_bytes
from p2.decisions import DecisionStore
from p2.generate import mock_generate
from p2.ui import build_ui, new_session, on_decide, on_summarize


@pytest.fixture
def session(tmp_path, cfg):
    store = DecisionStore(tmp_path)
    sid, paths = new_session(store, image_bytes(), cfg, "A2", mock_generate, tmp_path)
    return store, sid, paths


def test_new_session_creates_five_images(session):
    store, sid, paths = session
    assert sorted(paths) == ["V1", "V2", "V3", "V4", "V5"]
    s = store.load(sid)
    assert [v.seed for v in s.variations] == [1001, 1002, 1003, 1004, 1005]


def test_decide_enables_summary_only_at_five(session):
    store, sid, _ = session
    for k in range(1, 5):
        msg, ready = on_decide(store, sid, f"V{k}", "aceptada")
        assert not ready
    msg, ready = on_decide(store, sid, "V5", "rechazada", "no me gusta")
    assert ready and msg == "Decididas 5/5"


def test_summary_without_llm_is_verified_template(session):
    store, sid, _ = session
    for k in range(1, 6):
        on_decide(store, sid, f"V{k}", "aceptada" if k in (2, 3) else "rechazada")
    res = on_summarize(store, sid)
    assert res.verified and res.fallback and res.summary.aceptadas == ["V2", "V3"]
    saved = json.loads((store.root / sid / "summary.json").read_text(encoding="utf-8"))
    assert saved["verified"] is True


def test_build_ui_constructs(tmp_path, cfg):
    pytest.importorskip("gradio")
    demo = build_ui(DecisionStore(tmp_path), cfg)
    assert demo is not None


def test_new_session_records_actual_generation_params(tmp_path, cfg):
    """La app usa strength 0.55 (config app): debe quedar registrado ese valor, no el base 0.40."""
    store = DecisionStore(tmp_path)
    gen = {**cfg["generation"], "strength": cfg["app"]["strength"]}
    sid, _ = new_session(store, image_bytes(), cfg, cfg["app"]["method"], mock_generate, tmp_path, gen)
    s = store.load(sid)
    assert all(v.params["strength"] == 0.55 and v.method == "A2" for v in s.variations)
    assert [v.attribute for v in s.variations] == ["luz_calida", "luz_fria_estudio", "acuarela", "sonrisa", "bn_epoca"]


def test_ui_serves_session_files_outside_temp_with_allowed_paths(cfg, tmp_path):
    """Regresión del error en Colab: sesiones fuera de cwd/temp solo se sirven si se pasan en allowed_paths."""
    gradio = pytest.importorskip("gradio")
    client_mod = pytest.importorskip("gradio_client")
    import shutil
    from pathlib import Path
    from conftest import ROOT
    root = ROOT.parent / "_p2_test_sesiones_fuera_de_temp"  # ni en el cwd del proyecto ni en %TEMP%
    shutil.rmtree(root, ignore_errors=True)
    try:
        store = DecisionStore(root)
        demo = build_ui(store, cfg, out_dir=root)
        demo.launch(prevent_thread_lock=True, quiet=True, allowed_paths=[str(root)])
        img = tmp_path / "in.png"
        img.write_bytes(image_bytes())
        out = client_mod.Client(demo.local_url, verbose=False).predict(
            client_mod.handle_file(str(img)), "", True, api_name="/_gen")
        assert sum(isinstance(x, (str, dict)) for x in out) >= 5
        demo.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_img_html_embeds_png_inline(tmp_path):
    """Las variaciones se incrustan como data URI: no dependen de servir archivos (falla en el visor de Colab)."""
    import base64
    import re
    from PIL import Image
    from p2.ui import img_html
    p = tmp_path / "V1.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(p)
    html = img_html(p, "V1 · luz_calida")
    m = re.search(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"', html)
    assert m and base64.b64decode(m.group(1)) == p.read_bytes() and "V1 · luz_calida" in html
    assert "file=" not in html and "http" not in html


def test_full_ui_flow_returns_inline_images_and_json(cfg, tmp_path):
    """Recorrido completo contra el servidor gradio real: 5 imágenes incrustadas, decisiones y resumen en JSON."""
    pytest.importorskip("gradio")
    client_mod = pytest.importorskip("gradio_client")
    store = DecisionStore(tmp_path / "sesiones")
    demo = build_ui(store, cfg, out_dir=store.root)
    demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        c = client_mod.Client(demo.local_url, verbose=False)
        img = tmp_path / "in.png"
        img.write_bytes(image_bytes())
        out = c.predict(client_mod.handle_file(str(img)), "", True, api_name="/_gen")
        figs = [x for x in out if isinstance(x, str) and "data:image/png;base64," in x]
        assert len(figs) == 5 and "V3 · acuarela" in figs[2]
        decs_in = ["aceptada", "rechazada", "aceptada", "rechazada", "aceptada"]
        md, decs, status = c.predict(*decs_in, "", "motivo escrito tarde", "", "", "", api_name="/_sum")
        assert status == "Decididas 5/5" and decs["variations"][1]["reason"] == "motivo escrito tarde"
        sus_md = c.predict("", *([4, 2] * 5), api_name="/_sus")  # cuestionario SUS desde la app real
        assert "75.0 / 100" in sus_md and (store.root / "sus.jsonl").exists()
        assert "verificado" in md and [v["decision"] for v in decs["variations"]][:2] == ["aceptada", "rechazada"]
    finally:
        demo.close()


def test_on_submit_registers_screen_state_and_reports_missing(tmp_path, cfg):
    from p2.ui import on_submit
    store = DecisionStore(tmp_path)
    sid, _ = new_session(store, image_bytes(), cfg, "A2", mock_generate, tmp_path)
    with pytest.raises(ValueError, match="Faltan decisiones: V2, V4"):
        on_submit(store, sid, ["aceptada", None, "rechazada", None, "aceptada"], [""] * 5)
    s = on_submit(store, sid, ["aceptada", "aceptada", "aceptada", "rechazada", "rechazada"],
                  ["", "", "", "cambio de género", "  deformación  "])
    assert s.accepted() == ["V1", "V2", "V3"] and s.by_id("V5").reason == "deformación"
    n = len(store.events(sid))
    on_submit(store, sid, ["aceptada", "aceptada", "aceptada", "rechazada", "rechazada"],
              ["", "", "", "cambio de género", "deformación"])
    assert len(store.events(sid)) == n  # repetir sin cambios no duplica eventos


def test_browser_click_flow_records_late_reasons(cfg, tmp_path):
    """Navegador real (Chromium headless): clics en las opciones y motivo escrito DESPUÉS de elegir."""
    pytest.importorskip("gradio")
    sync_api = pytest.importorskip("playwright.sync_api")
    from conftest import ROOT
    store = DecisionStore(tmp_path / "sesiones")
    demo = build_ui(store, cfg, out_dir=store.root)
    demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        with sync_api.sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception as e:  # navegador no instalado en este equipo
                pytest.skip(f"Chromium no disponible: {e}")
            pg = b.new_page()
            pg.goto(demo.local_url)
            pg.set_input_files("input[type=file]", str(ROOT / "data" / "sfhq_t2i" / "SDXL_image_0035846.jpg"))
            pg.get_by_label("Confirmo que tengo derecho a usar esta imagen").check()
            pg.get_by_role("button", name="Generar 5 variaciones").click()
            pg.wait_for_selector("img[src^='data:image/png;base64']", timeout=60000)
            for k, d in enumerate(["aceptada", "aceptada", "aceptada", "rechazada", "rechazada"], start=1):
                pg.locator("label", has_text=d).nth(k - 1).click()
            box = pg.get_by_label("Motivo (opcional)").nth(3)
            box.fill("cambio de género")  # escrito después de elegir
            box.press("Tab")
            pg.wait_for_timeout(300)  # una persona tarda en ir al botón; gradio sincroniza el valor al salir
            pg.get_by_role("button", name="Generar resumen").click()
            pg.wait_for_selector("text=verificado", timeout=30000)  # el resumen ya terminó (no basta el contador)
            b.close()
        sid = next(store.root.iterdir()).name
        s = store.load(sid)
        assert s.accepted() == ["V1", "V2", "V3"] and s.rejected() == ["V4", "V5"]
        assert s.by_id("V4").reason == "cambio de género"
        assert (store.root / sid / "summary.json").exists()
    finally:
        demo.close()


def test_nsfw_variation_is_labelled_in_ui(cfg, tmp_path):
    """Si el filtro de seguridad bloquea una variación (imagen en negro), la app lo dice en su leyenda."""
    pytest.importorskip("gradio")
    client_mod = pytest.importorskip("gradio_client")
    from PIL import Image
    def gen_one(img, spec):  # V3 bloqueada, como en la sesión s-a8ed0d72 de Colab
        return (Image.new("RGB", (512, 512)), True) if spec["id"] == "V3" else mock_generate(img, spec)
    store = DecisionStore(tmp_path / "sesiones")
    demo = build_ui(store, cfg, out_dir=store.root, generate_one=gen_one)
    demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        img = tmp_path / "in.png"
        img.write_bytes(image_bytes())
        out = client_mod.Client(demo.local_url, verbose=False).predict(client_mod.handle_file(str(img)), "", True, api_name="/_gen")
        figs = [x for x in out if isinstance(x, str) and "data:image/png;base64," in x]
        assert "filtro de seguridad" in figs[2] and all("filtro de seguridad" not in f for i, f in enumerate(figs) if i != 2)
    finally:
        demo.close()
