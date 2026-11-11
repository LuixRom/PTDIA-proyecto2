"""Vitrina IA en la laptop (respaldo si no hay GPU en Colab/Kaggle). Usa los modelos ya descargados, sin internet.

Preparación (una vez):
    .venv\\Scripts\\python.exe -m pip install "gradio==6.15.2"
    .venv\\Scripts\\python.exe tools\\download_models.py --llm
Uso:
    .venv\\Scripts\\python.exe tools\\app_local.py            -> abre http://127.0.0.1:7860 en el navegador

Ajustes para una GPU de 4 GB: Stable Diffusion con descarga a CPU, 1 candidato por look (sin CLIP) y solo Qwen
como explicador, que se carga en la GPU únicamente mientras redacta el resumen (no caben juntos). Las sesiones se
guardan fuera del repositorio, en C:\\PTDIA\\sesiones_local (bórrelas al terminar: contienen fotos de personas).
"""
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def on_demand_llm(entry):
    """Cargador para vitrina_ui.LLMs: el modelo se carga en la GPU solo durante cada respuesta y se libera."""
    import torch

    from p2.llm import load_llm_entry

    def fn(prompt):
        generate, _, model = load_llm_entry(entry, "cuda")
        try:
            return generate(prompt)
        finally:
            del model, generate
            gc.collect()
            torch.cuda.empty_cache()
    return fn, f"{entry['repo']}@{str(entry['revision'])[:8]}", None


def main():
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit('Falta gradio en el .venv: .venv\\Scripts\\python.exe -m pip install "gradio==6.15.2"')
    from p2 import __version__, config, tryon, vitrina, vitrina_ui
    from p2.decisions import DecisionStore
    from p2.tempdb import TempDB
    cfg = config.load("config/vitrina.yaml")
    cfg["approaches"]["B"]["candidates"] = 1
    cfg["llms"] = {"qwen": cfg["llms"]["qwen"]}  # Phi (7,6 GB) no cabe en 4 GB
    print(f"Vitrina IA {__version__} (local) — cargando modelos…")
    seg, pose = tryon.Segmenter(cfg), tryon.PoseEstimator(cfg)
    pipe = tryon.load_inpaint(cfg, offload=True)
    engine = vitrina.Engine(cfg, seg, None,
                            lambda img, mask, spec, control=None: tryon.gen_b(pipe, img, mask, spec, cfg, control),
                            lambda img: tryon.nsfw_input(pipe, img), None, pose=pose.detect)
    from p2 import api_images
    ed = api_images.ApiEditor(**cfg["approaches"]["API"]["api"])
    if ed.available():
        engine.api = ed
        print(f"Enfoque API: disponible ({ed.model}; de pago, pide consentimiento en cada uso).")
    else:
        print("Enfoque API: no disponible (no hay clave en .env). La app usa solo el enfoque local.")
    store = DecisionStore(Path(os.environ.get("VITRINA_SESIONES", r"C:\PTDIA\sesiones_local")))
    db = TempDB(store.root / "agente_tmp.sqlite")
    demo = vitrina_ui.build_ui(store, cfg, engine, vitrina_ui.LLMs(cfg, on_demand_llm), db)
    print("Sesiones en", store.root)
    demo.launch(inbrowser=True, allowed_paths=[str(store.root)], show_error=True, **vitrina_ui.launch_kwargs())


if __name__ == "__main__":
    main()
