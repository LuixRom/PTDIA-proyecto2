"""Perfil de estilo, verificador de preferencias, ronda con estilo y exportación de las pruebas (modelos simulados)."""
import json

import pytest
from PIL import Image

from conftest import image_bytes
from p2 import api_images, config, estilo, vitrina
from p2.decisions import DecisionStore
from p2.ui import on_submit, on_sus
from p2.vitrina_ui import stub_llm_loader

PHOTO = image_bytes(size=(640, 640))
OPTS = dict(mode="ocasion", occasion="trabajo", linea="mujer", approach="B", seed_mode="prueba", own_photo=True,
            adult=True, participant="Ana Pérez")
DECS = ["aceptada", "aceptada", "rechazada", "aceptada", "rechazada"]
MOTIVO = "no me gustan los pantalones pitillo, me gustan los rectos o baggy"


@pytest.fixture(scope="module")
def vcfg():
    return config.load("config/vitrina.yaml")


@pytest.fixture
def ses(tmp_path, vcfg):
    store, eng = DecisionStore(tmp_path / "sesiones"), vitrina.Engine.mock(vcfg)
    sid, _ = vitrina.start_session(store, eng, PHOTO, OPTS)
    on_submit(store, sid, DECS, ["", "", MOTIVO, "", ""])
    return store, eng, sid


def test_every_garment_is_tagged_and_first_round_mixes_cuts(vcfg):
    ropa = {k: v for k, v in vcfg["catalogo"].items() if v.get("tipo") != "accesorio"}
    assert all(v.get(c) for v in ropa.values() for c in estilo.CAMPOS)
    assert estilo.etiquetas(vcfg, ["m_blusa_blanca", "x_no_existe"]) == ["color blanco", "corte regular", "estilo office casual"]
    for occ in vcfg["ocasiones"].values():
        for linea in ("mujer", "hombre"):
            cortes = {t for lk in occ[linea][:5] for i in lk if vcfg["catalogo"][i].get("parte") == "abajo"
                      for t in estilo.etiquetas(vcfg, [i], ("corte",))}
            assert len(cortes) >= 2, (occ["nombre"], linea, cortes)  # ronda 1: al menos dos cortes de pantalón


def test_catalog_accessories_go_with_clothes(vcfg):
    from p2.guardrails import GuardrailError
    with pytest.raises(GuardrailError, match="al menos una prenda de ropa"):
        vitrina.looks_for(vcfg, {"mode": "catalogo", "items": ["h_lentes_metal"]})
    looks, _ = vitrina.looks_for(vcfg, {"mode": "catalogo", "items": ["h_lentes_metal", "h_blazer_gris", "h_jean_recto"]})
    assert all(lk["accessories"] == ["h_lentes_metal"] and lk["items"] for lk in looks)


def test_deterministic_profile_cites_evidence(ses, vcfg):
    store, _, sid = ses
    p = estilo.perfil(store.load(sid), vcfg)
    no = {x["valor"]: x for x in p["no_quiero"]}
    assert "corte pitillo" in no and no["corte pitillo"]["evidencia"] == [{"look": "V3-r1", "decision": "rechazada"}]
    assert "corte baggy" in {x["valor"] for x in p["quiero"]}  # V2 (pantalón ancho) aceptado
    # recto: V1 aceptado y V5 rechazado -> 0, no aparece (solo las etiquetas; los motivos los lee el LLM)
    assert "corte recto" not in {x["valor"] for x in p["quiero"] + p["no_quiero"]}
    assert not {x["valor"] for x in p["quiero"]} & set(no)


@pytest.mark.parametrize("valor, esperado", [("corte pitillo", "no_quiero"), ("corte recto", "quiero"),
                                             ("corte baggy", "quiero"), ("color rojo", None)])
def test_reason_polarity(valor, esperado, vcfg):
    assert estilo.polaridad(MOTIVO, valor, vcfg) == esperado
    assert estilo.polaridad("muy formal", "estilo formal", vcfg) == "neutral"


def test_verifier_accepts_backed_and_rejects_unbacked_preferences(ses, vcfg):
    store, _, sid = ses
    s = store.load(sid)
    ok = {"preferencias": [{"valor": "corte recto", "tipo": "quiero", "look": "V3-r1"},       # motivo
                           {"valor": "corte baggy", "tipo": "quiero", "look": "V2-r1"}],     # etiqueta + aceptada
          "explicacion": "Prefieres los rectos, como dijiste en V3-r1."}
    errores, buenas = estilo.verificar_preferencias(ok, s, vcfg)
    assert errores == [] and [b["fuente"] for b in buenas] == ["motivo", "etiqueta"]
    malo = {"preferencias": [{"valor": "corte pitillo", "tipo": "quiero", "look": "V3-r1"},   # contradice el motivo
                             {"valor": "color rojo", "tipo": "quiero", "look": "V1-r1"},      # sin respaldo
                             {"valor": "brillos", "tipo": "quiero", "look": "V1-r1"},         # fuera del vocabulario
                             {"valor": "corte recto", "tipo": "quiero", "look": "V9-r1"}],    # look inexistente
            "explicacion": "Te encantó V7."}
    errores, buenas = estilo.verificar_preferencias(malo, s, vcfg)
    assert buenas == [] and len(errores) == 5


def test_explain_with_llm_and_fallback(ses, vcfg):
    store, _, sid = ses
    fn, name, _ = stub_llm_loader(vcfg["llms"]["qwen"])
    r = vitrina.deducir_estilo(store, sid, vcfg, fn, name)
    assert r["fuente"] == "llm" and r["verificado"] and "corte pitillo" in [p["valor"] for p in r["no_quiero"]]
    calls = []

    def mentiroso(prompt):
        calls.append(prompt)
        return '{"preferencias": [{"valor": "color rojo", "tipo": "quiero", "look": "V1-r1"}], "explicacion": "x"}'
    r = vitrina.deducir_estilo(store, sid, vcfg, mentiroso, "malo")
    assert r["fuente"] == "perfil determinista" and len(calls) == 3 and "REVISIÓN" in calls[1]
    assert "color rojo" not in [p["valor"] for p in r["quiero"]] and r["errores"]  # lo no respaldado se descarta
    ev = [e for e in store.events(sid) if e["type"] == "perfil"]
    assert len(ev) == 2 and len((store.root / sid / "estilos.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_reason_overrides_deduction(ses, vcfg):
    base = {"quiero": [{"valor": "corte pitillo", "puntos": 1, "fuente": "decisiones", "evidencia": []}], "no_quiero": []}
    out = estilo.combinar(base, [{"valor": "corte pitillo", "tipo": "no_quiero", "look": "V3-r1", "fuente": "motivo"}])
    assert out["quiero"] == [] and out["no_quiero"][0]["valor"] == "corte pitillo"
    out = estilo.combinar(base, [{"valor": "corte pitillo", "tipo": "no_quiero", "look": "V3-r1", "fuente": "etiqueta"}])
    assert out["quiero"][0]["valor"] == "corte pitillo" and out["no_quiero"] == []  # una etiqueta no anula el cálculo


def test_selection_excludes_no_and_ranks_by_yes(vcfg):
    mostrados = ["Blusa blanca de botones + Pantalón de vestir gris"]
    got = estilo.elegir(vcfg, "mujer", "trabajo", ["corte baggy"], ["corte pitillo", "color negro"], mostrados, 3)
    assert len(got) == 3 and all(why.startswith("porque") or why.startswith("para") for _, why in got)
    for lk, _ in got:
        t = set(estilo.etiquetas(vcfg, lk["items"]))
        assert not t & {"corte pitillo", "color negro"} and lk["name"] not in mostrados
    assert "corte baggy" in estilo.etiquetas(vcfg, got[0][0]["items"]) and "corte baggy" in got[0][1]
    assert len({tuple(sorted(lk["items"])) for lk, _ in got}) == 3  # sin ropa repetida


def test_selection_falls_back_to_adjusted_look(vcfg):
    todo = estilo.vocabulario(vcfg, "mujer")
    got = estilo.elegir(vcfg, "mujer", "trabajo", ["corte recto", "color azul"], [t for t in todo if t.startswith("estilo")],
                        [], 2)
    assert len(got) == 2 and all("ajustado a tu estilo" in lk["name"] for lk, _ in got)
    assert "straight-leg fit" in got[0][0]["text"] or "navy blue color" in got[0][0]["text"]


def test_style_round_regenerates_rejected_and_logs_profile(ses, vcfg):
    store, eng, sid = ses
    why = vitrina.ronda_con_estilo(store, eng, sid, ["corte recto", "corte baggy"], ["corte pitillo"], False,
                                   {"quiero": ["corte recto"], "no_quiero": ["corte pitillo"]})
    s = store.load(sid)
    assert sorted(why) == ["V3", "V5"] and s.round == 2
    assert s.by_id("V1").decision == "aceptada" and s.by_id("V3").decision is None
    nuevos = [s.by_id(v) for v in why]
    assert all(v.metrics["por_que"] and "corte pitillo" not in v.metrics["etiquetas"] for v in nuevos)
    h = vitrina.history(store, sid)[-1]
    assert h["kind"] == "ronda_estilo" and h["estilo"]["deducido"]["quiero"] == ["corte recto"]
    assert any(e["type"] == "round_estilo" and e["no_quiero"] == ["corte pitillo"] for e in store.events(sid))
    on_submit(store, sid, ["aceptada"] * 5, [""] * 5)
    m = vitrina.session_metrics(store, sid, vcfg)
    assert m["tasa_aceptacion_por_ronda"] == {"r1": 0.6, "r2": 1.0} and m["rondas_con_estilo"] == 1
    assert m["ronda_estilo_respeta_no_quiero_pct"] == 100.0 and m["ronda_estilo_coincide_quiero_pct"] == 100.0
    assert m["regeneradas_que_estaban_rechazadas_pct"] == 100.0
    assert len(estilo.decididos(s)) == 5 and len(estilo.decididos(store.load(sid))) == 7  # r1 archivadas + r2
    with pytest.raises(ValueError, match="Las 5"):
        vitrina.ronda_con_estilo(store, eng, sid, [], [])
    assert len(vitrina.ronda_con_estilo(store, eng, sid, ["corte baggy"], [], todas=True)) == 5


def test_new_rounds_work_with_the_api_approach(tmp_path, vcfg):
    """Regresión: las nuevas rondas reconstruyen el contexto (antes faltaba keep_mask y la API fallaba)."""
    store, eng = DecisionStore(tmp_path / "s"), vitrina.Engine.mock(vcfg)
    eng.api = lambda img, texto: (img.copy(), {"total_tokens": 1})
    sid, _ = vitrina.start_session(store, eng, PHOTO, {**OPTS, "approach": "API", "api_consent": True})
    on_submit(store, sid, DECS, [""] * 5)
    vitrina.new_round(store, eng, sid)
    on_submit(store, sid, DECS, [""] * 5)
    assert len(vitrina.ronda_con_estilo(store, eng, sid, ["corte recto"], [])) == 2


def test_api_size_follows_photo_orientation():
    assert api_images.size_for(Image.new("RGB", (512, 512))) == "1024x1024"
    assert api_images.size_for(Image.new("RGB", (512, 768))) == "1024x1536"
    assert api_images.size_for(Image.new("RGB", (900, 500))) == "1536x1024"


def test_export_tests_to_excel_and_json(ses, vcfg, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from tools import exportar_pruebas
    store, eng, sid = ses
    fn, name, _ = stub_llm_loader(vcfg["llms"]["qwen"])
    vitrina.deducir_estilo(store, sid, vcfg, fn, name)
    vitrina.ronda_con_estilo(store, eng, sid, ["corte recto"], ["corte pitillo"])
    on_submit(store, sid, ["aceptada"] * 5, [""] * 5)
    on_sus(store, sid, "Ana Pérez", [4, 2, 4, 2, 4, 2, 4, 2, 4, 2])
    out = exportar_pruebas.main([str(store.root), "--salida", str(tmp_path / "pruebas.xlsx")])
    wb = openpyxl.load_workbook(out)
    assert wb.sheetnames == ["resumen", "sesiones", "decisiones", "perfiles", "SUS"]
    filas = lambda h: list(wb[h].iter_rows(values_only=True))  # noqa: E731
    cab, fila = filas("sesiones")[:2]
    ses_row = dict(zip(cab, fila))
    assert ses_row["participante"] == "Ana Pérez" and ses_row["aceptacion_r1"] == 0.6 and ses_row["aceptacion_r2"] == 1.0
    assert len(filas("decisiones")) == 1 + 7 and len(filas("perfiles")) == 1 + 2 and filas("SUS")[1][-1] == 75.0
    resumen = {k: v for k, v in filas("resumen")[1:]}
    assert resumen["aceptación media ronda 1"] == 0.6 and resumen["aceptación media ronda 2"] == 1.0
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["sesiones"][0]["participante"] == "Ana Pérez"
    assert not any(str(c.value).endswith(".png") for ws in wb for row in ws.iter_rows() for c in row if c.value)


def test_ui_style_flow(tmp_path, vcfg):
    pytest.importorskip("gradio")
    client_mod = pytest.importorskip("gradio_client")
    from p2.vitrina_ui import LLMs, build_ui
    store = DecisionStore(tmp_path / "sesiones")
    demo = build_ui(store, vcfg, vitrina.Engine.mock(vcfg), LLMs(vcfg, stub_llm_loader))
    demo.launch(prevent_thread_lock=True, quiet=True)
    try:
        c = client_mod.Client(demo.local_url, verbose=False)
        img = tmp_path / "yo.png"
        img.write_bytes(PHOTO)
        c.predict(client_mod.handle_file(str(img)), True, True, "Ana", "mujer", "Por ocasión", "Trabajo (office casual)",
                  [], "", vcfg["approaches"]["B"]["name"], vcfg["llms"]["qwen"]["name"], True, False, api_name="/_go")
        reasons = ["", "", MOTIVO, "", ""]
        md, q, n, *_ = c.predict(vcfg["llms"]["qwen"]["name"], *DECS, *reasons, api_name="/_deducir")
        assert "Tu estilo" in md and "corte pitillo" in n["value"] and "corte recto" in q["value"]
        out = c.predict("rechazadas", q["value"], n["value"], *DECS, *reasons, api_name="/_estilo_ronda")
        assert any(isinstance(x, str) and x.startswith("Nueva ronda con tu estilo") for x in out)
        assert sum("Elegido porque" in x for x in out if isinstance(x, str)) >= 1
    finally:
        demo.close()
