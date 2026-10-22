"""Interfaz Gradio de aceptación/rechazo. La lógica vive en funciones puras (testeables sin navegador).

Uso local con datos simulados:  python -m p2.ui --mock
"""
import argparse
import tempfile
import uuid
from pathlib import Path

from . import config as cfgmod
from .decisions import DecisionStore
from .explainer import explain, template_summary
from .generate import cinco_variaciones, mock_generate, plan
from .preprocess import prepare
from .schemas import ExplainResult, Session, Summary, Variation
from .verifier import verificar


def img_html(path, caption):
    """Imagen incrustada (data URI). No depende de que gradio sirva archivos: en el visor de Colab
    (share=False) las URLs de archivo apuntan a localhost y el navegador no las carga."""
    import base64
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (f'<figure style="margin:0"><img src="data:image/png;base64,{b64}" alt="{caption}" '
            f'style="width:100%;border-radius:8px"/><figcaption style="text-align:center">{caption}</figcaption></figure>')


def new_session(store, image_bytes, cfg, method, generate_one, out_dir, gen=None):
    """gen: parámetros de generación realmente usados (p.ej. strength de la app); por defecto, los de la config."""
    img, sha = prepare(image_bytes, cfg["image_size"])
    sid = f"s-{uuid.uuid4().hex[:8]}"
    d = Path(out_dir) / sid
    d.mkdir(parents=True, exist_ok=True)
    gen = gen or cfg["generation"]
    variations, paths = [], {}
    for spec, out, nsfw in cinco_variaciones(img, plan(cfg, method), generate_one):
        p = d / f"{spec['id']}.png"
        out.save(p)
        paths[spec["id"]] = str(p)
        variations.append(Variation(id=spec["id"], method=method, attribute=spec["attribute"], prompt=spec["prompt"],
                                    seed=spec["seed"], image_file=p.name, metrics={"nsfw": nsfw},
                                    params={k: gen[k] for k in ("steps", "strength", "guidance_scale", "scheduler")}))
    store.create(Session(session_id=sid, input_sha256=sha, variations=variations))
    return sid, paths


def on_decide(store, sid, var_id, decision, reason=""):
    s = store.decide(sid, var_id, decision, reason)
    done = sum(v.decision is not None for v in s.variations)
    return f"Decididas {done}/5", s.complete()


def on_submit(store, sid, decisions, reasons):
    """Registra las 5 decisiones y motivos TAL COMO ESTÁN EN PANTALLA al pedir el resumen.
    No depende de los eventos de clic (en Colab no siempre llegaban) y recoge motivos escritos después de elegir."""
    missing = [f"V{k}" for k, d in enumerate(decisions, start=1) if d not in ("aceptada", "rechazada")]
    if missing:
        raise ValueError(f"Faltan decisiones: {', '.join(missing)}")
    s = store.load(sid)
    for k, (d, r) in enumerate(zip(decisions, reasons), start=1):
        v, r = s.by_id(f"V{k}"), (r or "").strip()
        if (v.decision, v.reason) != (d, r):  # solo lo que cambió: events.jsonl no se llena de duplicados
            s = store.decide(sid, f"V{k}", d, r)
    return s


def on_summarize(store, sid, llm=None, llm_name="plantilla"):
    s = store.load(sid)
    if llm is None:  # sin LLM: plantilla determinista, marcada como tal
        data = template_summary(s)
        errs = verificar(data, s)
        res = ExplainResult(summary=Summary(**data), verified=not errs, errors=errs, attempts=0, fallback=True, llm=llm_name)
    else:
        res = explain(s, llm, verificar, llm_name=llm_name)
    out = store.root / sid / "summary.json"
    out.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    return res


def build_ui(store, cfg, method="A2", generate_one=mock_generate, out_dir=None, llm=None, llm_name="plantilla", gen=None):
    import gradio as gr
    out_dir = out_dir or store.root
    with gr.Blocks(title="Proyecto 2 — 5 variaciones") as demo:
        sid = gr.State(None)
        gr.Markdown("## Sube una imagen, revisa 5 variaciones y acepta o rechaza cada una\n"
                    "*Las variaciones son imágenes generadas por IA.* No subas fotos de terceros sin su consentimiento.")
        with gr.Row():
            inp = gr.Image(type="filepath", label="Imagen de entrada")
            consent = gr.Checkbox(label="Confirmo que tengo derecho a usar esta imagen")
        gen_btn = gr.Button("Generar 5 variaciones")
        imgs, radios, reasons = [], [], []
        with gr.Row():
            for k in range(1, 6):
                with gr.Column():
                    imgs.append(gr.HTML(f"<p style='text-align:center'>V{k}</p>"))
                    radios.append(gr.Radio(["aceptada", "rechazada"], label=f"Decisión V{k}"))
                    reasons.append(gr.Textbox(label="Motivo (opcional)", lines=1))
        status = gr.Markdown("Decididas 0/5")
        sum_btn = gr.Button("Generar resumen")
        summary_md = gr.Markdown()
        decisions_view = gr.JSON(label="Decisiones registradas")

        def _gen(path, ok):
            if not ok:
                raise gr.Error("Debes confirmar el consentimiento antes de continuar.")
            if not path:
                raise gr.Error("Sube una imagen.")
            s, paths = new_session(store, Path(path).read_bytes(), cfg, method, generate_one, out_dir, gen)
            vs = {v.id: v for v in store.load(s).variations}
            # el filtro de seguridad de SD devuelve la imagen en negro: se avisa en vez de dejarla sin explicación
            nsfw = " · ⚠️ bloqueada por el filtro de seguridad (imagen en negro); puede rechazarla"
            figs = [img_html(paths[f"V{k}"], f"V{k} · {vs[f'V{k}'].attribute}" + (nsfw if vs[f"V{k}"].metrics.get("nsfw") else ""))
                    for k in range(1, 6)]
            # nueva sesión: se limpian decisiones y motivos que la pantalla aún mostraba de la anterior
            return [s, *figs, *[gr.update(value=None)] * 5, *[""] * 5, "Decididas 0/5", "", None]

        gen_btn.click(_gen, [inp, consent], [sid, *imgs, *radios, *reasons, status, summary_md, decisions_view])

        for k in range(1, 6):
            def _dec(s, d, _k=k):  # contador informativo; el registro definitivo lo hace on_submit
                if s is None or d is None:
                    return gr.update()
                return on_decide(store, s, f"V{_k}", d, None)[0]  # el motivo se registra en on_submit
            radios[k - 1].change(_dec, [sid, radios[k - 1]], [status])

        def _sum(s, *vals):
            if s is None:
                raise gr.Error("Primero genere las 5 variaciones.")
            try:
                on_submit(store, s, vals[:5], vals[5:])
            except ValueError as e:
                raise gr.Error(str(e))
            res = on_summarize(store, s, llm, llm_name)
            badge = "✅ verificado" if res.verified else "⚠️ NO verificado: " + "; ".join(res.errors)
            origin = "plantilla determinista" if res.fallback else f"LLM {res.llm}"
            return (f"**Resumen** ({origin}, {badge})\n\n{res.summary.resumen}", store.load(s).model_dump(mode="json"),
                    "Decididas 5/5")

        sum_btn.click(_sum, [sid, *radios, *reasons], [summary_md, decisions_view, status])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="backend simulado (sin modelos)")
    a = ap.parse_args()
    if not a.mock:
        raise SystemExit("Solo el modo --mock está disponible localmente; la generación real se ejecuta en Colab.")
    store = DecisionStore(Path(tempfile.mkdtemp(prefix="p2_sessions_")))
    print("Sesiones en:", store.root)
    build_ui(store, cfgmod.load()).launch()


if __name__ == "__main__":
    main()
