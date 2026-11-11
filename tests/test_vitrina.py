from pathlib import Path
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from conftest import image_bytes
from p2 import config, guardrails, tryon, vitrina
from p2.decisions import DecisionStore
from p2.guardrails import GuardrailError
from p2.ui import on_submit

ROOT_DIR = Path(__file__).resolve().parent.parent
PHOTO = image_bytes(size=(640, 640))
OPTS = dict(mode="ocasion", occasion="reunion", linea="mujer", approach="B", seed_mode="prueba",
            own_photo=True, adult=True)


@pytest.fixture(scope="module")
def vcfg():
    return config.load("config/vitrina.yaml")


@pytest.fixture
def env(tmp_path, vcfg):
    return DecisionStore(tmp_path / "sesiones"), vitrina.Engine.mock(vcfg)


def test_config_occasions_have_five_looks_and_known_items(vcfg):
    assert vcfg["num_variations"] == 5 and len(vcfg["instruction_variants"]) == 5
    for occ in vcfg["ocasiones"].values():
        for linea in ("mujer", "hombre"):
            assert len(occ[linea]) >= 5
            assert all(i in vcfg["catalogo"] and vcfg["catalogo"][i]["linea"] == linea for look in occ[linea] for i in look)
    assert set(vcfg["llms"]) == {"qwen", "phi"} and set(vcfg["approaches"]) == {"A", "B", "API"}
    assert vcfg["generation"]["safety_checker"] is True


@pytest.mark.parametrize("text", ["quiero verme sin ropa", "Make her THINNER", "lencería roja", "una casaca Nike",
                                  "ropa para niño", "undress"])
def test_guardrail_blocks(text, vcfg):
    with pytest.raises(GuardrailError):
        guardrails.check_text(text, vcfg)


@pytest.mark.parametrize("text", ["una casaca verde de jean", "camino al trabajo con blazer", "vestido rojo elegante"])
def test_guardrail_allows(text, vcfg):
    assert guardrails.check_text(text, vcfg) == text


def test_consent_and_age_required(env):
    store, eng = env
    for own, adult in ((False, True), (True, False)):
        with pytest.raises(GuardrailError, match="mayor de edad"):
            vitrina.start_session(store, eng, PHOTO, {**OPTS, "own_photo": own, "adult": adult})
    assert not store.root.exists() or not any(store.root.iterdir())  # nada generado ni guardado


def test_mask_never_covers_face_or_hair():
    lab = tryon.MockSegmenter().labels(Image.new("RGB", (512, 512)))
    mask, frac = tryon.mask_from_labels(lab, tryon.MOCK_ID2LABEL, ["Upper-clothes", "Left-arm", "Right-arm"],
                                        ["Face", "Hair"], dilate_px=15)
    m = np.asarray(mask) > 0
    assert 0.2 < frac < 0.6 and not m[np.isin(lab, [11, 2])].any()
    assert tryon.face_box(lab, tryon.MOCK_ID2LABEL) is not None and tryon.count_faces_labels(lab, tryon.MOCK_ID2LABEL) == 1


def test_two_faces_rejected(env):
    store, eng = env

    class TwoFaces(tryon.MockSegmenter):
        def labels(self, img):
            lab = super().labels(img)
            lab[10:120, 10:120] = 11
            return lab
    eng.segmenter = TwoFaces()
    with pytest.raises(GuardrailError, match="2 rostros"):
        vitrina.start_session(store, eng, PHOTO, OPTS)


def test_session_round_restore_and_history(env, vcfg):
    store, eng = env
    sid, paths = vitrina.start_session(store, eng, PHOTO, OPTS)
    s = store.load(sid)
    assert len(paths) == 5 and [v.seed for v in s.variations] == [1001, 1002, 1003, 1004, 1005]
    assert all(v.metrics["face_ssim"] == 1.0 for v in s.variations)  # B: el rostro queda intacto
    first = {v.id: v.attribute for v in s.variations}

    with pytest.raises(ValueError, match="Decide las 5"):
        vitrina.new_round(store, eng, sid)
    on_submit(store, sid, ["aceptada", "rechazada", "aceptada", "rechazada", "aceptada"], ["", "color", "", "", ""])
    changed = vitrina.new_round(store, eng, sid)
    s = store.load(sid)
    assert sorted(changed) == ["V2", "V4"] and s.round == 2
    assert s.by_id("V1").decision == "aceptada" and s.by_id("V2").decision is None  # aceptadas se conservan
    assert s.by_id("V2").attribute not in first.values() and s.by_id("V2").seed == 1102  # alternativa no mostrada

    vitrina.restore(store, eng, sid, 1)
    s = store.load(sid)
    assert {v.id: v.attribute for v in s.variations} == first and [v.seed for v in s.variations][:2] == [1001, 1002]
    h = vitrina.history(store, sid)
    assert [x["kind"] for x in h] == ["inicial", "nueva_ronda", "restaurada"] and h[1]["regenerated"] == ["V2", "V4"]
    assert [e["type"] for e in store.events(sid)].count("round") == 1


def test_production_seeds_vary_and_test_seeds_repeat(env):
    store, eng = env
    a, _ = vitrina.start_session(store, eng, PHOTO, {**OPTS, "seed_mode": "produccion"})
    b, _ = vitrina.start_session(store, eng, PHOTO, {**OPTS, "seed_mode": "produccion"})
    c, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    seeds = lambda x: [v.seed for v in store.load(x).variations]  # noqa: E731
    assert seeds(a) != seeds(b) and seeds(c) == [1001, 1002, 1003, 1004, 1005]


def test_catalog_and_instruction_modes(env, vcfg):
    store, eng = env
    sid, _ = vitrina.start_session(store, eng, PHOTO, {**OPTS, "mode": "catalogo", "items": ["m_vestido_rojo", "m_camisa_celeste"]})
    assert [v.items for v in store.load(sid).variations][:3] == [["m_vestido_rojo"], ["m_camisa_celeste"], ["m_vestido_rojo"]]
    with pytest.raises(GuardrailError, match="entre 1 y 5"):
        vitrina.start_session(store, eng, PHOTO, {**OPTS, "mode": "catalogo", "items": []})
    eng.translate = lambda t: "a green denim jacket"
    sid, _ = vitrina.start_session(store, eng, PHOTO, {**OPTS, "mode": "instruccion", "instruction": "casaca verde"})
    vs = store.load(sid).variations
    assert len({v.prompt for v in vs}) == 5 and "green denim jacket" in vs[0].prompt
    eng.translate = lambda t: "naked person"  # una traducción maliciosa también se bloquea
    with pytest.raises(GuardrailError):
        vitrina.start_session(store, eng, PHOTO, {**OPTS, "mode": "instruccion", "instruction": "casaca verde"})


def test_identity_guardrail_retries_once(env):
    store, eng = env
    eng.gen_b = lambda img, mask, spec, **kw: (Image.new("RGB", img.size, (255, 0, 0)), False)  # destruye el rostro
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    v = store.load(sid).by_id("V1")
    assert v.metrics["attempts"] == 2 and v.metrics["identity_ok"] is False


def test_approach_a_recomposes_face_and_background(env):
    store, eng = env
    eng.gen_a = lambda img, spec: (Image.new("RGB", img.size, (255, 0, 0)), False)  # img2img cambia todo
    sid, _ = vitrina.start_session(store, eng, PHOTO, {**OPTS, "approach": "A"})
    v = store.load(sid).by_id("V1")
    assert v.metrics["attempts"] == 1 and v.metrics["face_ssim"] > 0.95  # el rostro vuelve a ser el original


def test_nsfw_false_positive_retried_with_new_seed(env):
    store, eng = env
    eng.cfg = {**eng.cfg, "guardrails": {**eng.cfg["guardrails"], "output_filter": "bloquear"}}  # modo estricto
    calls = []

    def flaky(img, mask, spec, **kw):  # primera vez bloqueada (falso positivo), segunda bien
        calls.append(spec["seed"])
        return tryon.mock_b(img, mask, spec)[0], len(calls) == 1
    eng.gen_b = flaky
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    v = store.load(sid).by_id("V1")
    assert calls[:2] == [1001, 1001 + 7919] and v.metrics["nsfw"] is False and v.metrics["attempts"] == 2


def test_summary_metrics_export_and_delete(env, vcfg):
    store, eng = env
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    on_submit(store, sid, ["aceptada", "rechazada", "aceptada", "aceptada", "rechazada"], [""] * 5)
    from p2.vitrina_ui import stub_llm_loader
    fn, name, _ = stub_llm_loader(vcfg["llms"]["phi"])
    res = vitrina.summarize(store, sid, fn, name)
    assert res.verified and not res.fallback and res.summary.aceptadas == ["V1", "V3", "V4"]
    m = vitrina.session_metrics(store, sid, vcfg)
    assert m["tasa_aceptacion"] == 0.6 and m["resumenes_verificados_llm"] == 1 and m["rondas"] == 1
    z = vitrina.export_looks(store, sid, vcfg)
    with zipfile.ZipFile(z) as f:
        assert sorted(f.namelist()) == ["LEEME.txt", "V1.png", "V3.png", "V4.png", "looks.json"]
        assert json.loads(f.read("looks.json"))[0]["prendas"]
    assert vitrina.delete_session(store, sid) and not (store.root / sid).exists()


def test_vitrina_ui_full_flow(tmp_path, vcfg):
    pytest.importorskip("gradio")
    client_mod = pytest.importorskip("gradio_client")
    from p2.vitrina_ui import LLMs, build_ui, stub_llm_loader
    store = DecisionStore(tmp_path / "sesiones")
    demo = build_ui(store, vcfg, vitrina.Engine.mock(vcfg), LLMs(vcfg, stub_llm_loader))
    demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        c = client_mod.Client(demo.local_url, verbose=False)
        img = tmp_path / "yo.png"
        img.write_bytes(PHOTO)
        out = c.predict(client_mod.handle_file(str(img)), True, True, "", "mujer", "Por ocasión", "Evento", [], "",
                        vcfg["approaches"]["B"]["name"], vcfg["llms"]["qwen"]["name"], True, False, api_name="/_go")
        cards = [x for x in out if isinstance(x, str) and "data:image/png;base64," in x]
        assert len(cards) == 5 and "Vestido largo rojo" in cards[0]  # (enfoque B, elegido explícitamente)
        decs = ["aceptada", "rechazada", "aceptada", "aceptada", "aceptada"]
        out = c.predict(*decs, "", "muy formal", "", "", "", api_name="/_again")
        assert "V2" in next(x for x in out if isinstance(x, str) and x.startswith("Nueva ronda"))
        md, status, *_ = c.predict(vcfg["llms"]["phi"]["name"], *(["aceptada"] * 5), *([""] * 5), api_name="/_summ")
        assert "verificado" in md and "stub:Phi" in md
        assert c.predict(*(["aceptada"] * 5), *([""] * 5), api_name="/_save").endswith(".zip")
    finally:
        demo.close()


def test_vitrina_notebook_current_and_safe(tmp_path):
    import re
    from conftest import ROOT
    from tools.build_vitrina_notebook import build
    nb = ROOT / "vitrina_colab.ipynb"
    assert build(tmp_path / "nb.ipynb").read_text(encoding="utf-8") == nb.read_text(encoding="utf-8"), \
        "Ejecute tools/build_vitrina_notebook.py"
    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    for flag in ("AUTORIZO_DESCARGAR_MODELOS", "AUTORIZO_APP", "ENLACE_PUBLICO", "AUTORIZO_BENCHMARK",
                 "CONFIRMO_CONSENTIMIENTO_FOTOS", "CONFIRMO_BORRAR_DATOS"):
        assert re.search(rf"^{flag} = False", code, re.M), flag
    assert not re.search(r"[!%]pip|\bpip\s+install|subprocess|os\.system", code)
    assert all(c.get("outputs", []) == [] for c in cells if c["cell_type"] == "code")


def test_tall_full_body_photo_is_padded_not_cropped():
    import io
    from p2.preprocess import prepare
    im = Image.new("RGB", (400, 1000), (255, 255, 255))
    im.paste((200, 30, 30), (150, 20, 250, 120))  # "cabeza" arriba de todo
    buf = io.BytesIO(); im.save(buf, "PNG")
    out, _ = prepare(buf.getvalue(), 512, fit="pad")
    a = np.asarray(out)
    assert out.size == (512, 768) and (a[:80, 200:330] == [200, 30, 30]).all(axis=-1).any()  # retrato; cabeza entera
    assert tuple(a[5, 5]) == (255, 255, 255)  # relleno con el color del fondo
    sq, _ = prepare(image_bytes(size=(640, 600)), 512, fit="pad")
    assert sq.size == (512, 512)  # fotos no verticales: cuadradas
    cropped, _ = prepare(buf.getvalue(), 512)  # el recorte de la versión 0 la habría perdido
    assert not (np.asarray(cropped) == [200, 30, 30]).all(axis=-1).any()


def test_flagged_input_warns_but_continues(env):
    store, eng = env
    eng.nsfw_input = lambda img: True
    sid, paths = vitrina.start_session(store, eng, PHOTO, OPTS)
    assert len(paths) == 5 and store.load(sid).options["input_nsfw_flag"] is True


def test_browser_like_then_save_without_summary(tmp_path, vcfg):
    """Caso reportado: marcar «Me gusta» y guardar SIN pedir el resumen debe registrar y exportar."""
    pytest.importorskip("gradio")
    pw = pytest.importorskip("playwright.sync_api")
    from p2.vitrina_ui import LLMs, build_ui, stub_llm_loader
    store = DecisionStore(tmp_path / "sesiones")
    demo = build_ui(store, vcfg, vitrina.Engine.mock(vcfg), LLMs(vcfg, stub_llm_loader))
    demo.launch(prevent_thread_lock=True, quiet=True)
    img = tmp_path / "yo.png"
    img.write_bytes(PHOTO)
    try:
        with pw.sync_playwright() as p:
            try:
                b = p.chromium.launch()
            except Exception as e:  # navegador no instalado
                pytest.skip(f"chromium no disponible: {e}")
            pg = b.new_page()
            pg.goto(demo.local_url)
            pg.set_input_files("input[type=file]", str(img))
            pg.get_by_label("La foto es mía").check()
            pg.get_by_label("Soy mayor de edad").check()
            pg.get_by_text("Mujer", exact=True).click()
            pg.get_by_role("button", name="Probarme 5 looks").click()
            pg.wait_for_timeout(3000)
            pg.get_by_text("Me gusta", exact=True).nth(0).click()
            pg.get_by_text("Me gusta", exact=True).nth(2).click()
            pg.wait_for_timeout(1500)
            sid = next(d.name for d in store.root.iterdir() if d.is_dir())
            assert store.load(sid).accepted() == ["V1", "V3"]  # registrado al hacer clic
            pg.get_by_role("button", name="Guardar mis looks (ZIP)").click()
            pg.wait_for_timeout(2000)
            assert list((store.root / sid).glob("mis_looks_*.zip"))
            pg.get_by_role("button", name="Aceptar las 5").click()
            pg.wait_for_timeout(1500)
            assert store.load(sid).accepted() == ["V1", "V2", "V3", "V4", "V5"]
            b.close()
    finally:
        demo.close()


def test_portrait_session_end_to_end(env):
    store, eng = env
    tall = image_bytes(size=(400, 900))
    sid, paths = vitrina.start_session(store, eng, tall, OPTS)
    assert Image.open(paths["V1"]).size == (512, 768)
    on_submit(store, sid, ["rechazada"] * 5, [""] * 5)
    vitrina.new_round(store, eng, sid)
    vitrina.restore(store, eng, sid, 1)
    assert store.load(sid).round == 3


def test_zone_by_look(vcfg):
    assert vitrina.zone_of(vcfg, ["m_blusa_blanca"]) == "superior"
    assert vitrina.zone_of(vcfg, ["m_blusa_blanca", "m_pantalon_gris"]) == "completo"
    assert vitrina.zone_of(vcfg, []) == "completo"  # instrucción escrita: conjunto completo


def test_region_inpainting_works_on_enlarged_crop_and_keeps_outside():
    img = Image.new("RGB", (512, 768), (200, 200, 200))
    mask = Image.new("L", (512, 768), 0)
    mask.paste(255, (200, 300, 300, 500))  # ropa pequeña en una foto de cuerpo entero
    sizes = []

    def run(crop, m, size, c):
        sizes.append(size)
        assert crop.size == size and m.size == size and max(size) == 768 and all(v % 8 == 0 for v in size)
        return Image.new("RGB", size, (0, 0, 255)), False
    out, nsfw = tryon.inpaint_region(run, img, mask, 768)
    a = np.asarray(out)
    assert sizes and not nsfw and out.size == img.size
    assert tuple(a[400, 250]) == (0, 0, 255) and tuple(a[50, 50]) == (200, 200, 200)  # dentro cambia, fuera no


def test_persistently_blocked_look_shows_card_not_black(env, vcfg):
    store, eng = env
    eng.cfg = {**eng.cfg, "guardrails": {**eng.cfg["guardrails"], "output_filter": "bloquear"}}  # modo estricto
    calls = []

    def always_blocked(img, mask, spec, **kw):
        calls.append(spec["seed"])
        return Image.new("RGB", img.size, (0, 0, 0)), True
    eng.gen_b = always_blocked
    sid, paths = vitrina.start_session(store, eng, PHOTO, OPTS)
    v = store.load(sid).by_id("V1")
    assert v.metrics["attempts"] == 1 + vcfg["guardrails"]["nsfw_retries"] and v.metrics["nsfw"] is True
    assert np.asarray(Image.open(paths["V1"])).mean() > 150  # recuadro claro, no la silueta negra


def test_openpose_control_has_skeleton_but_no_photo_texture():
    img = Image.new("RGB", (512, 768), (255, 255, 255))
    ctrl = np.asarray(tryon.openpose_image(tryon.mock_pose(img), img.size))
    assert ctrl.shape == (768, 512, 3) and ctrl.any()          # se dibuja el esqueleto
    assert (ctrl.reshape(-1, 3) == 0).all(axis=1).mean() > 0.9  # fondo negro: sin textura ni marcas de agua de la foto

def test_sleeves_conditional_arms_and_hands(env, vcfg):
    store, eng = env
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    d = store.root / sid
    img = Image.open(d / "input.png")
    lab = tryon.MockSegmenter().labels(img)
    arms = np.isin(lab, [14, 15])
    hands = tryon.hands_from_pose(lab, tryon.MOCK_ID2LABEL, tryon.mock_pose(img))
    skin_hands = hands & tryon.skin(img)  # de la zona de la mano solo se protege la piel (la tela de la manga se repinta)
    assert hands.any() and hands[int(img.height * .9):].any() and not hands[:int(img.height * .8)].any()  # solo junto a la muñeca
    for zona in ("superior", "completo"):
        short = np.asarray(Image.open(d / f"mask_{zona}.png")) > 0
        long_ = np.asarray(Image.open(d / f"mask_{zona}_mangas.png")) > 0
        assert not short[arms].any()                                            # manga corta: brazos intactos
        assert long_[arms & ~hands].mean() > 0.9 and not long_[skin_hands].any()  # manga larga: brazos sí, manos nunca
        assert not long_[np.isin(lab, [11])].any()                              # rostro nunca
    assert (d / "control.png").exists()  # esqueleto para ControlNet openpose
    assert vitrina.long_sleeves(vcfg, ["h_casaca_jean"]) and not vitrina.long_sleeves(vcfg, ["m_vestido_floral"])
    assert vitrina.long_sleeves(vcfg, [], "a green denim jacket") and not vitrina.long_sleeves(vcfg, [], "a red top")

def test_without_pose_arms_are_never_repainted(env):
    store, eng = env
    eng.pose = None
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    assert not list((store.root / sid).glob("mask_*_mangas.png"))  # sin pose: siempre la máscara segura

def test_best_of_candidates_by_clip_and_exact_restore(env, vcfg):
    store, eng = env

    class FakeClip:  # «más parecido al texto» = más azul
        def txt_emb(self, text):  # la sonda «estampado» apunta al rojo: el candidato rojo queda penalizado
            return np.array([1.0, 0.0, 0.0]) if "stripes" in text else np.array([0.0, 0.0, 1.0])

        def img_emb(self, im):
            v = np.asarray(im.convert("RGB"), dtype=float).mean(axis=(0, 1))
            return v / (np.linalg.norm(v) + 1e-9)
    calls = []

    def by_seed(img, mask, spec, **kw):  # semilla base -> rojo; candidato 2 -> azul
        calls.append(spec["seed"])
        color = (0, 0, 255) if spec["seed"] > 100000 else (255, 0, 0)
        return Image.composite(Image.new("RGB", img.size, color), img, mask), False
    eng.clip, eng.gen_b = FakeClip(), by_seed
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    v1 = store.load(sid).by_id("V1")
    n = vcfg["approaches"]["B"]["candidates"]
    assert n == 2 and v1.seed == 1001 + 104729 and len(v1.metrics["candidates_clip"]) == 2  # eligió el azul
    calls.clear()
    vitrina.restore(store, eng, sid, 1)
    assert calls[:1] == [1001 + 104729] and len(calls) == 5  # restaurar: 1 generación por look, la semilla elegida


def test_hair_core_protected_edge_repaintable():
    lab = np.zeros((200, 200), dtype=np.int64)
    lab[60:200, 40:160] = 4          # ropa
    lab[20:60, 80:120] = 11          # rostro
    lab[10:140, 60:80] = 2           # mechón de cabello que cae sobre el hombro (encima de la ropa)
    lab[60:120, 120:130] = 14        # brazo que sube hasta la cara (mano en la mejilla)
    lab[120:200, 160:180] = 14       # otro tramo del brazo, lejos de la cara
    id2 = tryon.MOCK_ID2LABEL
    strict, _ = tryon.mask_from_labels(lab, id2, ["Upper-clothes"], ["Face", "Hair"], dilate_px=9)
    core, _ = tryon.mask_from_labels(lab, id2, ["Upper-clothes"], ["Face", "Hair"], dilate_px=9, core_only={"Hair": 5})
    s, c = np.asarray(strict) > 0, np.asarray(core) > 0
    hair = lab == 2
    assert not s[hair].any() and c[hair].any()            # el borde del mechón ahora se repinta
    assert not c[70:130, 66:74].any()                     # el núcleo del mechón sigue protegido
    assert not c[lab == 11].any()                         # el rostro nunca


def test_layered_prompt_and_upper_only_bottom(vcfg):
    lk = vitrina._look(vcfg, ["h_blazer_gris", "m_blusa_blanca"])
    assert lk["text"].startswith("solid grey blazer") and "blouse" not in lk["text"]  # solo la prenda de encima
    assert lk["items"] == ["h_blazer_gris", "m_blusa_blanca"]                                  # la tienda muestra las dos
    assert lk["text"].endswith(vcfg["approaches"]["B"]["upper_only_suffix"])          # solo arriba: parte de abajo neutra
    assert "bottoms" not in vitrina._look(vcfg, ["m_blusa_blanca", "m_pantalon_gris"])["text"]  # con pantalón no


def test_hand_halo_of_white_background_is_not_protected():
    img = Image.new("RGB", (100, 100), (250, 250, 250))
    img.paste((205, 150, 120), (40, 40, 60, 60))  # piel
    keep = tryon.not_background(img)
    assert keep[50, 50] and not keep[5, 5]


def test_production_picks_random_looks_test_mode_fixed(vcfg):
    import random
    base = dict(mode="ocasion", occasion="trabajo", linea="mujer")
    fixed = [lk["name"] for lk in vitrina.looks_for(vcfg, {**base, "seed_mode": "prueba"}, random.Random(1))[0]]
    assert fixed == [lk["name"] for lk in vitrina.looks_for(vcfg, {**base, "seed_mode": "prueba"}, random.Random(2))[0]]
    picks = {tuple(lk["name"] for lk in vitrina.looks_for(vcfg, {**base, "seed_mode": "produccion"}, random.Random(s))[0])
             for s in range(6)}
    assert len(picks) > 1  # producción: cada prueba muestra una selección distinta del catálogo
    looks, alt = vitrina.looks_for(vcfg, {**base, "seed_mode": "produccion"}, random.Random(3))
    assert len(looks) == 5 and len(alt) >= 1 and not {x["name"] for x in looks} & {x["name"] for x in alt}


def test_default_output_filter_only_logs(env):
    store, eng = env
    eng.gen_b = lambda img, mask, spec, **kw: (tryon.mock_b(img, mask, spec)[0], True)  # el filtro «salta» siempre
    sid, paths = vitrina.start_session(store, eng, PHOTO, OPTS)
    v = store.load(sid).by_id("V1")
    assert v.metrics["nsfw"] is False and v.metrics["nsfw_alertas"] >= 1 and v.metrics["attempts"] == 1  # se muestra
    assert vitrina.session_metrics(store, sid, eng.cfg)["alertas_filtro_sin_bloqueo"] >= 5


def test_skin_detector_keeps_skin_not_striped_fabric():
    img = Image.new("RGB", (60, 20), (255, 255, 255))
    img.paste((198, 134, 103), (0, 0, 20, 20))   # piel
    img.paste((20, 30, 80), (20, 0, 40, 20))     # tela azul marino
    sk = tryon.skin(img)
    assert sk[:, :20].all() and not sk[:, 20:].any() and not sk[:, 40:].any()


def test_outer_garments_are_closed(vcfg):
    assert "fully buttoned" in vitrina._look(vcfg, ["h_blazer_gris", "m_blusa_blanca"])["text"]
    assert "buttoned" not in vitrina._look(vcfg, ["m_camisa_celeste"])["text"]


def test_hand_region_never_reaches_back_toward_elbow():
    lab = np.zeros((200, 200), dtype=np.int64)
    lab[40:160, 90:110] = 14                      # brazo vertical: hombro arriba, mano abajo
    k = np.zeros((17, 3))
    k[7] = (100, 80, 1)                          # codo
    k[9] = (100, 140, 1)                         # muñeca (mano hacia abajo)
    hand = tryon.hands_from_pose(lab, tryon.MOCK_ID2LABEL, k)
    assert hand[150:160, 90:110].any() and not hand[40:130, :].any()  # la mano sí; brazo y hombro no


def test_bottom_garment_is_always_described(vcfg):
    lk = vitrina._look(vcfg, ["m_camisa_celeste", "m_jean_azul"])
    assert "jeans" in lk["text"] and lk["name"].endswith("Jean azul recto")


def test_shorts_only_when_legs_visible(vcfg):
    import random
    base = dict(mode="ocasion", occasion="casual", linea="hombre", seed_mode="prueba")
    names = lambda o: [x["name"] for x in sum(vitrina.looks_for(vcfg, o, random.Random(1)), [])]  # noqa: E731
    assert any("Short" in n for n in names({**base, "has_pants": False}))
    assert not any("Short" in n for n in names({**base, "has_pants": True}))


def test_accessories_kept_out_of_clothing_text(vcfg):
    look = vitrina._look(vcfg, ["m_blusa_blanca", "m_pantalon_gris", "m_lentes_negros", "m_collar_perlas"])
    assert look["items"] == ["m_blusa_blanca", "m_pantalon_gris"]
    assert look["accessories"] == ["m_lentes_negros", "m_collar_perlas"]
    assert "eyeglasses" not in look["text"] and "Lentes" in look["name"]


def test_negative_prompt_is_returned(vcfg):
    from p2 import tryon
    spec = {"prompt": "a white shirt"}
    assert "extra hands" in tryon.negative_for(spec, vcfg)
    assert tryon.negative_for({**spec, "accessory": True}, vcfg) == vcfg["accesorios"]["negative_prompt"]


def test_accessory_masks_from_pose():
    import numpy as np
    from p2 import tryon
    id2 = {0: "Background", 2: "Hair", 11: "Face", 9: "Left-shoe", 10: "Right-shoe"}
    lab = np.zeros((200, 100), int)
    lab[10:50, 30:70] = 11          # rostro
    lab[180:195, 30:45] = 9          # zapatos
    lab[180:195, 55:70] = 10
    kps = [(50, 30, .9), (60, 25, .9), (40, 25, .9), (76, 30, .9), (24, 30, .9), (80, 70, .9), (20, 70, .9)] + [(0, 0, 0)] * 10
    m = tryon.accessory_masks(lab, id2, kps, {})
    assert set(m) == {"lentes", "collar"}
    assert np.asarray(m["lentes"])[25, 50] and not np.asarray(m["lentes"])[45, 50]
    lab[:, :] = np.where(lab == 11, 11, 2)  # cabello alrededor de la cara: tapa las orejas
    lab[180:195, 30:45] = 0
    lab[180:195, 55:70] = 0
    m = tryon.accessory_masks(lab, id2, kps, {})
    assert "lentes" in m


def test_accessories_applied_in_session(env):
    store, eng = env
    gen, calls = eng.gen_b, []

    def spy(img, mask, spec, control=None):
        calls.append(bool(spec.get("accessory")))
        return gen(img, mask, spec, control)
    eng.gen_b = spy
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)  # reunión mujer: cada look lleva accesorios
    s = store.load(sid)
    con = [v for v in s.variations if v.items != [] and "+ Lentes" in v.attribute or "+ Collar" in v.attribute]
    assert 1 <= len(con) <= 2  # accesorios en pocos looks, no en todos
    for v in con:
        m = v.metrics
        assert set(m["accesorios_aplicados"]) | set(m["accesorios_omitidos"])
        assert not set(m["accesorios_aplicados"]) & set(m["accesorios_omitidos"])
    assert calls.count(False) >= 5  # la ropa sigue generándose aparte


def test_dresses_dropped_when_photo_has_pants(vcfg):
    import random
    base = dict(mode="ocasion", occasion="evento", linea="mujer", seed_mode="prueba")
    names = lambda o: [x["name"] for x in sum(vitrina.looks_for(vcfg, o, random.Random(1)), [])]  # noqa: E731
    assert any("Vestido" in n for n in names({**base, "has_pants": False}))
    got = vitrina.looks_for(vcfg, {**base, "has_pants": True}, random.Random(1))[0]
    assert len(got) == 5 and not any("Vestido" in x["name"] for x in got)


def test_only_some_looks_carry_accessories(vcfg):
    for occ in vcfg["ocasiones"].values():
        for linea in ("mujer", "hombre"):
            n = sum(any(vcfg["catalogo"][i].get("tipo") == "accesorio" for i in look) for look in occ[linea][:5])
            assert 1 <= n <= 2


def test_restore_background_keeps_garment_and_clears_ghost():
    import numpy as np
    from p2 import tryon
    orig = Image.new("RGB", (40, 40), (220, 220, 225))
    gen = orig.copy()
    gen.paste((205, 205, 212), (0, 0, 10, 40))   # sombra tenue sobre el fondo
    gen.paste((20, 40, 160), (30, 0, 40, 40))    # prenda que sobresale del fondo
    bg = np.ones((40, 40), bool)
    out = np.asarray(tryon.restore_background(orig, gen, bg, 28))
    assert abs(int(out[20, 3, 0]) - 220) <= 3 and out[20, 36, 2] > 100 and out[20, 36, 0] < 60


def test_halo_of_old_silhouette_is_replaced_by_clean_background():
    import numpy as np
    from p2 import tryon
    a = np.full((60, 60, 3), (225, 222, 230), np.uint8)
    a[:, 20:40] = (60, 90, 160)                       # jean ancho original
    orig = Image.fromarray(a)
    bg_old = np.ones((60, 60), bool); bg_old[:, 20:40] = False
    g = a.copy(); g[:, 20:26] = (245, 245, 248); g[:, 26:40] = (10, 20, 200)   # halo claro + prenda nueva más angosta
    bg_new = np.ones((60, 60), bool); bg_new[:, 26:40] = False
    out = np.asarray(tryon.paste_clean_background(Image.fromarray(g), tryon.clean_background(orig, bg_old), bg_new, bg_old))
    assert abs(int(out[30, 22, 0]) - 225) <= 4 and out[30, 32, 2] == 200


def test_wrist_zone_frees_the_wrist_but_not_the_fingers():
    import numpy as np
    from p2 import tryon
    kps = [(0, 0, 0)] * 17
    kps[7], kps[9] = (50, 100, .9), (50, 200, .9)    # codo y muñeca: antebrazo de 100 px hacia abajo
    z = tryon.wrist_zone((300, 100), kps)
    assert z[200, 50] and z[185, 50] and not z[240, 50] and not z[150, 50]


def test_design_fonts_are_bundled_so_the_page_looks_the_same_offline():
    from p2 import vitrina_theme as vt
    assert vt.CSS.count("data:font/woff2;base64,") == 2 and "Big Shoulders Display" in vt.CSS
    assert "fonts.googleapis.com" not in vt.CSS and "@import" not in vt.CSS     # nada se pide a internet
    assert (ROOT_DIR / "p2/assets/fonts/OFL-NOTA.txt").exists()


def test_card_shows_decision_state_and_ui_hides_advanced_settings(tmp_path, vcfg):
    from p2 import vitrina_theme as vt
    from p2.vitrina_ui import LLMs, build_ui, stub_llm_loader
    img = tmp_path / "x.png"
    img.write_bytes(PHOTO)
    card = vt.look_card(0, "V1", "Blusa blanca + Pantalón gris", 1, img)
    assert "est-ok'>Aceptado" in card and "est-no'>Descartado" in card and "title='Blusa blanca + Pantalón gris'" in card
    demo = build_ui(DecisionStore(tmp_path / "s"), vcfg, vitrina.Engine.mock(vcfg), LLMs(vcfg, stub_llm_loader))
    cfgtxt = str(demo.get_config_file()["components"])
    assert "Ajustes avanzados" in cfgtxt and "look-col" in cfgtxt and "Cómo generar tus looks" in cfgtxt
    assert ":has(input[value=\"aceptada\"]:checked)" in vt.CSS and "body.generando" in vt.CSS
    assert "#consent-api" in vt.CSS and "#approach input" in vt.CSS


def test_ejemplo_de_pose_incluido():
    from p2 import vitrina_theme as vt
    h = vt.ejemplo_html()
    assert "data:image/jpeg;base64," in h and "Cómo tomar tu foto" in h
    assert (ROOT_DIR / "p2/assets/ejemplos/pose_ejemplo.jpg").exists()
