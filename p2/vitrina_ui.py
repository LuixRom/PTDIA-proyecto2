"""Interfaz web de Vitrina IA (Gradio). La lógica vive en p2/vitrina.py; aquí solo se conecta la pantalla.

Uso local sin modelos:  python -m p2.vitrina_ui --mock
En Colab: celda 5 de vitrina_colab.ipynb (modelos reales, enlace público opcional).
"""
import argparse
import gc
import tempfile
from pathlib import Path

from . import __version__
from . import config as cfgmod
from . import sus, vitrina
from .decisions import DecisionStore
from .guardrails import GuardrailError
from . import vitrina_theme as vt
from .ui import on_submit, on_sus

CSS = vt.CSS

MODE_LABELS = {"Por ocasión": "ocasion", "Prendas del catálogo": "catalogo", "Escribir mi idea": "instruccion"}


class LLMs:
    """Mantiene cargado un solo LLM a la vez (la T4 no tiene memoria para los dos junto a los generadores)."""

    def __init__(self, cfg, loader):
        self.cfg, self.loader = cfg, loader
        self.key, self.fn, self.label, self.model = None, None, None, None

    def get(self, key):
        if key != self.key:
            self.unload()
            self.fn, self.label, self.model = self.loader(self.cfg["llms"][key])
            self.key = key
        return self.fn, self.label

    def unload(self):
        if self.model is not None:
            del self.model
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        self.key, self.fn, self.label, self.model = None, None, None, None


def stub_llm_loader(entry):
    """LLM simulado para el modo --mock: responde con la plantilla (el verificador lo valida igual)."""
    import json
    import re

    def fn(prompt):
        if prompt.startswith("Traduce"):
            return prompt.rsplit("Descripción:", 1)[1].strip()
        datos = json.loads(re.search(r"<datos>\s*(\[.*?\])\s*</datos>", prompt, re.S).group(1))
        acc = [d["id"] for d in datos if d["decision"] == "aceptada"]
        rej = [d["id"] for d in datos if d["decision"] == "rechazada"]
        txt = " ".join(f"{d['id']} ({d['atributo']}) fue {d['decision']}." for d in datos)
        return json.dumps({"aceptadas": acc, "rechazadas": rej, "resumen": txt}, ensure_ascii=False)
    return fn, f"stub:{entry['name']}", None


def _cards(store, sid, captions_only=False):
    s = store.load(sid)
    out = []
    for k, v in enumerate(s.variations):
        aviso = ""
        if v.metrics.get("nsfw"):
            aviso = "Bloqueada por el filtro de contenido."
        elif v.metrics.get("identity_ok") is False:
            aviso = "El rostro cambió: no se conservó tu identidad."
        out.append(vt.look_card(k, v.id, v.attribute, v.round, store.root / sid / v.image_file, aviso))
    return out


def build_ui(store, cfg, engine, llms, db=None):
    import gradio as gr

    occ = {v["nombre"]: k for k, v in cfg["ocasiones"].items()}
    # el enfoque API solo aparece si hay clave (engine.api): sin ella la app funciona igual, 100 % local
    app_names = {cfg["approaches"][k]["name"]: k for k in cfg.get("app_approaches", ["A", "B"])
                 if k != "API" or engine.api is not None}
    llm_names = {v["name"]: k for k, v in cfg["llms"].items()}

    def catalog_choices(linea):
        return [(v["nombre"], k) for k, v in vitrina.catalog_for(cfg, linea).items()]

    with gr.Blocks(title="Vitrina IA · Probador virtual") as demo:
        sid = gr.State(None)
        gr.HTML(vt.hero_html(__version__))
        gr.HTML(vt.ejemplo_html())
        with gr.Row(elem_id="form-row"):
            with gr.Column(scale=4):
                gr.HTML(vt.sec(1, "Tu foto"))
                photo = gr.Image(type="filepath", label="Tu foto", height=340)
                gr.HTML("<p class='nota'>De frente, solo tú, con los brazos relajados a los costados y sin marcas de agua.</p>")
                own = gr.Checkbox(label="La foto es mía (o tengo permiso de la persona)")
                adult = gr.Checkbox(label="Soy mayor de edad")
                alias = gr.Textbox(label="Alias (opcional, no uses tu nombre real)", max_lines=1)
            with gr.Column(scale=6):
                gr.HTML(vt.sec(2, "Elige qué probarte"))
                # sin valor por defecto: el cliente elige la línea (no se adivina el género a partir de la foto)
                linea = gr.Radio([("Mujer", "mujer"), ("Hombre", "hombre")], value=None, label="Línea de la tienda (elige una)")
                mode = gr.Radio(list(MODE_LABELS), value="Por ocasión", label="Cómo elegir los looks")
                occasion = gr.Dropdown(list(occ), value=next(iter(occ)), label="Ocasión")
                items = gr.CheckboxGroup([], label="Prendas (1 a 5) — primero elige la línea", visible=False)
                instr = gr.Textbox(label="Describe la prenda (p. ej. «una casaca verde de jean»)", visible=False, max_lines=2)
                # Las dos maneras de generar se ven siempre: es lo que se quiere mostrar (intermedio local y API)
                approach = gr.Radio(list(app_names), value=next(iter(app_names)), label="Cómo generar tus looks",
                                    info=" | ".join(cfg["approaches"][k]["description"] for k in app_names.values()),
                                    elem_id="approach")
                if "API" not in app_names.values():
                    gr.HTML("<p class='nota'>La otra manera, por API de imágenes, aparece cuando hay una clave en el "
                            "archivo .env.</p>")
                # La casilla se muestra u oculta con CSS según el enfoque elegido (sin evento ni cola: no puede colgarse).
                # El servidor sigue exigiendo el consentimiento al generar (vitrina.start_session).
                api_ok = gr.Checkbox(label="Acepto que mi foto se envíe a OpenAI (API de pago, fuera de mi equipo)",
                                     visible="API" in app_names.values(), elem_id="consent-api")
                with gr.Accordion("Ajustes avanzados: modelo que explica y semillas", open=False):
                    llm = gr.Radio(list(llm_names), value=next(iter(llm_names)), label="Modelo que explica tu elección")
                    test_mode = gr.Checkbox(label="Modo prueba (semillas fijas, resultados reproducibles)", value=False)
                go = gr.Button("Probarme 5 looks", variant="primary", size="lg")
                gr.HTML("<p class='generando-msg' role='status'>Midiendo tus cinco looks… puede tardar un par de minutos.</p>"
                        "<p class='nota-envio'>Tu foto no se guarda: se usa una copia reducida mientras dura la sesión y "
                        "puedes borrarla cuando quieras (pestaña Privacidad, más abajo).</p>")

        gr.HTML(vt.sec(3, "Tus cinco looks"))
        imgs, radios, reasons = [], [], []
        with gr.Row(equal_height=True, elem_id="looks-row"):
            for k in range(1, 6):
                with gr.Column(min_width=180, elem_classes=["look-col"]):
                    imgs.append(gr.HTML(vt.empty_card(k)))
                    radios.append(gr.Radio([("Me gusta", "aceptada"), ("No me gusta", "rechazada")], label=f"Decisión del look {k}",
                                           elem_classes=["decide"]))
                    reasons.append(gr.Textbox(label="¿Por qué? (opcional)", lines=1, max_lines=2))
        with gr.Row(elem_id="acciones"):
            all_yes = gr.Button("Aceptar las 5")
            all_no = gr.Button("Rechazar las 5")
            again = gr.Button("Nueva ronda con las rechazadas")
            summ = gr.Button("Explicar mi elección", variant="primary")
        status = gr.Markdown()
        summary_md = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Mis looks"):
                gallery = gr.Gallery(label="Looks aceptados", columns=5, height=400, object_fit="contain")
                looks_md = gr.Markdown()
                save_btn = gr.Button("Guardar mis looks (ZIP)")
                save_file = gr.File(label="Descarga", interactive=False, height=70, visible=False)  # aparece al guardar
            with gr.Tab("Historial"):
                hist_df = gr.Dataframe(headers=["ronda", "tipo", "enfoque", "semillas", "looks", "GPU s"], interactive=False)
                with gr.Row():
                    restore_round = gr.Dropdown([], label="Volver a la configuración de la ronda")
                    restore_btn = gr.Button("Restaurar")
            with gr.Tab("Métricas de la sesión"):
                metrics_json = gr.JSON()
            with gr.Tab("Registro de decisiones"):
                decisions_json = gr.JSON()
            with gr.Tab("Privacidad"):
                gr.Markdown("Tu foto no se guarda: solo una copia reducida (512×512, sin metadatos) mientras dura la sesión, "
                            "para poder generar nuevas rondas. Puedes borrarla ahora.")
                del_btn = gr.Button("Borrar mi sesión", variant="stop")
                del_md = gr.Markdown()
        with gr.Accordion("Cuestionario de usabilidad (SUS) — al terminar", open=False):
            gr.Markdown("1 = totalmente en desacuerdo · 5 = totalmente de acuerdo")
            sus_items = [gr.Radio([1, 2, 3, 4, 5], label=f"{i}. {q}") for i, q in enumerate(sus.ITEMS, start=1)]
            sus_btn = gr.Button("Enviar cuestionario")
            sus_md = gr.Markdown()
        gr.Markdown("Las imágenes son **simulaciones generadas con IA** (llevan la etiqueta «Simulación IA»); la prenda real "
                    "puede variar. No se permiten fotos de terceros sin permiso, de menores ni pedidos para desvestir o "
                    "cambiar el cuerpo.", elem_id="aviso")

        # ----------------------------------------------------------- callbacks
        def side_panels(s):
            hist = vitrina.history(store, s)
            rows = [[h["round"], h["kind"], h["specs"][0]["approach"], " ".join(str(x["seed"]) for x in h["specs"]),
                     " | ".join(x["look"] for x in h["specs"]), h.get("gpu_s")] for h in hist]
            acc = vitrina.accepted_looks(store, s, cfg)
            looks_txt = "\n".join(f"- **{lk['id']} · {lk['look']}** — prendas: {', '.join(lk['prendas']) or 'descritas por ti'} "
                                  "· *ver en la tienda (catálogo de demostración)*" for lk in acc) or "Aún no aceptaste looks."
            return (rows, gr.update(choices=[h["round"] for h in hist], value=None),
                    [(lk["imagen"], f"{lk['id']} · {lk['look']}") for lk in acc], looks_txt,
                    vitrina.session_metrics(store, s, cfg), store.load(s).model_dump(mode="json"))

        side_out = [hist_df, restore_round, gallery, looks_md, metrics_json, decisions_json]

        def full_refresh(s, msg):
            cards = _cards(store, s)
            st = store.load(s)
            return [s, *cards, *[v.decision for v in st.variations], *[v.reason for v in st.variations], msg, "",
                    *side_panels(s)]

        full_out = [sid, *imgs, *radios, *reasons, status, summary_md, *side_out]

        def _mode(m):
            k = MODE_LABELS[m]
            return gr.update(visible=k == "ocasion"), gr.update(visible=k == "catalogo"), gr.update(visible=k == "instruccion")

        mode.change(_mode, [mode], [occasion, items, instr])
        linea.change(lambda l: gr.update(choices=catalog_choices(l), value=[]), [linea], [items])

        def _go(path, own_v, adult_v, alias_v, linea_v, mode_v, occ_v, items_v, instr_v, app_v, llm_v, test_v,
                api_v=False):
            if not path:
                raise gr.Error("Sube tu foto.")
            if linea_v not in ("mujer", "hombre"):
                raise gr.Error("Elige la línea de la tienda (Mujer u Hombre) para ver prendas de esa línea.")
            opts = dict(mode=MODE_LABELS[mode_v], occasion=occ[occ_v], linea=linea_v, items=items_v, instruction=instr_v,
                        approach=app_names[app_v], seed_mode="prueba" if test_v else "produccion",
                        llm=llm_names[llm_v], participant=alias_v, own_photo=own_v, adult=adult_v,
                        api_consent=bool(api_v))
            if opts["mode"] == "instruccion":
                fn, _ = llms.get(opts["llm"])
                engine.translate = lambda t: fn(vitrina.TRANSLATE_PROMPT.format(texto=t)).strip().splitlines()[0]
            try:
                s, _ = vitrina.start_session(store, engine, Path(path).read_bytes(), opts)
            except (GuardrailError, ValueError) as e:
                raise gr.Error(str(e))
            msg = "Elige «Me gusta» o «No me gusta» en cada look (o usa los botones de paquete)."
            if store.load(s).options.get("input_nsfw_flag"):
                msg = ("Aviso: el filtro de contenido marcó tu foto (suele pasar con hombros o piernas a la vista). Se "
                       "continuó; cada look generado también se filtra. " + msg)
            return full_refresh(s, msg)

        _mide_on = "() => { document.body.classList.add('generando'); }"
        _mide_off = "() => { document.body.classList.remove('generando'); }"
        go.click(None, None, None, js=_mide_on)  # solo visual: la cinta del encabezado empieza a «medir»
        go.click(_go, [photo, own, adult, alias, linea, mode, occasion, items, instr, approach, llm, test_mode, api_ok],
                 full_out).then(None, None, None, js=_mide_off)
        def register(s, vals):
            """Registra al instante lo que hay en pantalla (decisiones tomadas y sus motivos), aunque falten
            looks por decidir: así Mis looks, Métricas y Registro coinciden siempre con lo que el usuario ve."""
            st = store.load(s)
            for k, (d, r) in enumerate(zip(vals[:5], vals[5:]), start=1):
                v, r = st.by_id(f"V{k}"), (r or "").strip()
                if d in ("aceptada", "rechazada") and (v.decision, v.reason) != (d, r):
                    st = store.decide(s, f"V{k}", d, r)
            done = sum(v.decision is not None for v in st.variations)
            return f"Decididas {done}/5." + ("" if done == 5 else " Puedes guardar tus looks aceptados en cualquier momento.")

        def _sync(s, *vals):
            if s is None:
                return gr.update(), *[gr.update()] * len(side_out)
            return register(s, vals), *side_panels(s)

        # .input (no .change): solo clics y textos del usuario, no los valores que pone la app al refrescar
        for comp in radios:
            comp.input(_sync, [sid, *radios, *reasons], [status, *side_out])
        for comp in reasons:
            comp.blur(_sync, [sid, *radios, *reasons], [status, *side_out])

        def _bulk(value):
            def fn(s, *reason_vals):
                if s is None:
                    raise gr.Error("Primero genera tus 5 looks.")
                return [value] * 5 + [register(s, [value] * 5 + list(reason_vals)), *side_panels(s)]
            return fn

        all_yes.click(_bulk("aceptada"), [sid, *reasons], [*radios, status, *side_out])
        all_no.click(_bulk("rechazada"), [sid, *reasons], [*radios, status, *side_out])

        def _submit(s, vals):
            if s is None:
                raise gr.Error("Primero genera tus 5 looks.")
            try:
                on_submit(store, s, vals[:5], vals[5:])
            except ValueError as e:
                raise gr.Error(str(e))

        def _again(s, *vals):
            _submit(s, vals)
            try:
                changed = vitrina.new_round(store, engine, s)
            except (GuardrailError, ValueError) as e:
                raise gr.Error(str(e))
            return full_refresh(s, f"Nueva ronda: se regeneraron {', '.join(sorted(changed))}; tus looks aceptados se mantienen.")

        again.click(None, None, None, js=_mide_on)
        again.click(_again, [sid, *radios, *reasons], full_out).then(None, None, None, js=_mide_off)

        def _summ(s, llm_v, *vals):
            _submit(s, vals)
            fn, label = llms.get(llm_names[llm_v])
            res = vitrina.summarize(store, s, fn, label, db)
            badge = "verificado contra tu registro" if res.verified else "NO verificado: " + "; ".join(res.errors)
            origin = "plantilla de respaldo (el modelo no superó la verificación)" if res.fallback else label
            return (f"**Tu elección explicada** ({origin}, {badge})\n\n{res.summary.resumen}",
                    "Decisiones registradas.", *side_panels(s))

        summ.click(_summ, [sid, llm, *radios, *reasons], [summary_md, status, *side_out])

        def _restore(s, r):
            if s is None or r is None:
                raise gr.Error("Elige una ronda del historial.")
            vitrina.restore(store, engine, s, r)
            return full_refresh(s, f"Se restauró la configuración de la ronda {r} (mismos looks y semillas).")

        restore_btn.click(_restore, [sid, restore_round], full_out)

        def _save(s, *vals):
            if s is None:
                raise gr.Error("Primero genera tus looks.")
            register(s, vals)  # lo que está marcado en pantalla, aunque no se haya pedido el resumen
            try:
                return str(vitrina.export_looks(store, s, cfg))
            except ValueError as e:
                raise gr.Error(str(e) + " Marca «Me gusta» en al menos un look.")

        # _save devuelve la ruta (la API la usa tal cual); el recuadro de descarga aparece cuando ya hay archivo
        save_btn.click(_save, [sid, *radios, *reasons], [save_file]).then(
            lambda f: gr.update(visible=bool(f)), [save_file], [save_file], api_name=False)

        def _delete(s):
            if s is None:
                return None, "No hay sesión activa."
            vitrina.delete_session(store, s)
            return None, "Sesión borrada: foto, imágenes y registros eliminados."

        del_btn.click(_delete, [sid], [sid, del_md])

        def _sus(s, al, *answers):
            try:
                score = on_sus(store, s, al, answers, db)
            except ValueError as e:
                raise gr.Error(str(e))
            return f"Gracias. Puntuación SUS: **{score:.1f} / 100**."

        sus_btn.click(_sus, [sid, alias, *sus_items], [sus_md])
    return demo


def launch_kwargs():
    import gradio as gr
    return {"css": vt.CSS, "theme": vt.theme()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="modelos simulados (sin GPU)")
    a = ap.parse_args()
    if not a.mock:
        raise SystemExit("Localmente solo existe el modo --mock; los modelos reales se cargan en Colab.")
    cfg = cfgmod.load("config/vitrina.yaml")
    store = DecisionStore(Path(tempfile.mkdtemp(prefix="vitrina_")))
    print("Sesiones en:", store.root)
    build_ui(store, cfg, vitrina.Engine.mock(cfg), LLMs(cfg, stub_llm_loader)).launch(**launch_kwargs())


if __name__ == "__main__":
    main()
