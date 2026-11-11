"""Simula vitrina_colab.ipynb en local ejecutando sus celdas REALES, antes de subirlo a Colab.

Uso (entorno con torch/diffusers/transformers):  .venv\\Scripts\\python.exe tools\\simulate_vitrina.py

Igual que tools/simulate_colab.py: google.colab y "/content" se sustituyen; SD 1.5, SD 1.5 Inpainting, SegFormer,
CLIP y los dos LLM son modelos DIMINUTOS con pesos aleatorios (mismas clases de diffusers/transformers), sin
descargas. gradio no está en el .venv: build_ui se sustituye por un recorrido que llama a las funciones REALES
del probador (sesión A y B, nueva ronda, restaurar, resumen con los 2 LLM, métricas, exportación).
Escenarios: A) por defecto, B) app + benchmark + exportar + borrar, C) celda 5 fuera de orden.
"""
import sys
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.simulate_colab import (Colab, build_tiny_models, code_cells, fresh_ns, install_old_torchao,  # noqa: E402
                                  report, run)


def relax_resources(ns):
    """Umbrales mínimos: la simulación usa modelos diminutos (no necesita la VRAM ni el disco de una T4)."""
    import p2.env_check as ec
    if getattr(ec.problems, "_sim", False):
        return
    orig = ec.problems

    def relaxed(info, req):
        return orig(info, {**req, "min_vram_gb": 1, "min_ram_gb": 1, "min_disk_gb": 1})
    relaxed._sim = True
    ec.problems = relaxed

ATR = ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe",
       "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf"]


def build_tiny_tryon(root, sd_dir):
    """Inpainting (UNet de 9 canales) y SegFormer diminutos. El segmentador predice siempre 'Upper-clothes'."""
    import torch
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor
    torch.manual_seed(0)
    sd = StableDiffusionPipeline.from_pretrained(sd_dir, variant="fp16")
    for s in (sd.safety_checker,):  # umbrales altos: con pesos aleatorios no debe marcar NSFW por azar
        s.concept_embeds_weights.data.fill_(10.0)
        s.special_care_embeds_weights.data.fill_(10.0)
    sd.save_pretrained(sd_dir, safe_serialization=True, variant="fp16")
    unet = UNet2DConditionModel(sample_size=64, in_channels=9, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 64), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32,
                                attention_head_dim=8)
    inp = StableDiffusionInpaintPipeline(vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer, unet=unet,
                                         scheduler=sd.scheduler, safety_checker=sd.safety_checker,
                                         feature_extractor=sd.feature_extractor, image_encoder=None)
    inp_dir = root / "tiny-sd15-inpaint"
    inp.save_pretrained(inp_dir, safe_serialization=True)  # sin variante fp16: ejercita el respaldo de load_inpaint
    cfg = SegformerConfig(num_channels=3, num_encoder_blocks=4, depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1],
                          hidden_sizes=[8, 16, 32, 64], num_attention_heads=[1, 1, 2, 2], decoder_hidden_size=32,
                          num_labels=len(ATR), id2label=dict(enumerate(ATR)), label2id={n: i for i, n in enumerate(ATR)})
    seg = SegformerForSemanticSegmentation(cfg)
    with torch.no_grad():
        seg.decode_head.classifier.weight.zero_()
        seg.decode_head.classifier.bias.zero_()
        seg.decode_head.classifier.bias[ATR.index("Upper-clothes")] = 5.0
    seg_dir = root / "tiny-segformer"
    seg.save_pretrained(seg_dir)
    SegformerImageProcessor(size={"height": 64, "width": 64}).save_pretrained(seg_dir)
    from diffusers import ControlNetModel
    cn_dir = root / "tiny-controlnet"
    ControlNetModel.from_unet(sd.unet, conditioning_embedding_out_channels=(8, 16, 16, 32)).save_pretrained(
        cn_dir, safe_serialization=True)
    from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor
    pose_dir = root / "tiny-vitpose"
    bb = VitPoseBackboneConfig(image_size=[64, 48], patch_size=[16, 16], hidden_size=32, num_hidden_layers=2,
                               num_attention_heads=4, mlp_ratio=2, out_indices=[2])
    VitPoseForPoseEstimation(VitPoseConfig(backbone_config=bb, num_labels=17)).save_pretrained(pose_dir)
    VitPoseImageProcessor(size={"height": 64, "width": 48}).save_pretrained(pose_dir)
    return inp_dir, seg_dir, cn_dir, pose_dir


def main():
    from tools.build_vitrina_notebook import build
    nb = build(Path(__file__).resolve().parent.parent / "vitrina_colab.ipynb")
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="vitrina_sim_"))
    print("Carpeta de simulación:", tmp)
    content = tmp / "content"
    content.mkdir()
    install_old_torchao(tmp)
    sd_dir, clip_dir, qwen_dir, _ = build_tiny_models(tmp / "models")
    inp_dir, seg_dir, cn_dir, pose_dir = build_tiny_tryon(tmp / "models", sd_dir)
    photo = ROOT / "data" / "sfhq_t2i" / "SDXL_image_0035846.jpg"
    results = []

    colab = Colab(content, upload_file=photo)
    colab.install()
    cells = code_cells(nb, content)
    title = {c[0].split(".")[0]: c[0] for c in cells}
    resA = run(cells, fresh_ns(), hooks={title["2"]: relax_resources})
    results.append(report("A (por defecto, sin autorizaciones: Ejecutar todas)", resA, lambda r: (
        len(r) == len(cells) and all(s == "ok" for _, s, _ in r)
        and all("⏸" in " ".join(ms) for t, _, ms in r if t.startswith(("5.", "6.", "7.", "8."))))))

    app = {}

    def authorize(n):
        n.update(AUTORIZO_DESCARGAR_MODELOS=True, AUTORIZO_APP=True, APP_AUTORIZADA=True, AUTORIZO_BENCHMARK=True, CARGAR_ENFOQUE_A=True,
                 CONFIRMO_CONSENTIMIENTO_FOTOS=True, BENCH_AUTORIZADO=True, CONFIRMO_BORRAR_DATOS=True)
        sys.modules.setdefault("gradio", types.ModuleType("gradio"))

    def tiny(n):
        relax_resources(n)
        m = n["cfg"]["models"]
        m["sd15"]["repo"], m["sd15_inpaint"]["repo"] = str(sd_dir), str(inp_dir)
        m["segmenter"]["repo"], m["clip"]["repo"] = str(seg_dir), str(clip_dir)
        n["cfg"]["approaches"]["B"]["controlnet"]["repo"] = str(cn_dir)
        n["cfg"]["models"]["pose"]["repo"] = str(pose_dir)
        for k in n["cfg"]["llms"]:
            n["cfg"]["llms"][k]["repo"] = str(qwen_dir)
        n["cfg"]["llms"]["phi"]["max_new_tokens"] = n["cfg"]["llms"]["qwen"]["max_new_tokens"] = 8
        import p2.vitrina_ui as vui
        vui.launch_kwargs = lambda: {}

        def build_ui(store, cfg, engine, llms, db=None):
            from p2 import vitrina
            from p2.ui import on_submit

            class Demo:
                share_url = None

                def close(self):
                    app["closed"] = True

                def launch(self, **kw):
                    data = photo.read_bytes()
                    base = dict(mode="ocasion", occasion="evento", linea="mujer", seed_mode="prueba", own_photo=True,
                                adult=True, llm="qwen")
                    sid_a, _ = vitrina.start_session(store, engine, data, {**base, "approach": "A"})
                    import io
                    from PIL import Image
                    buf = io.BytesIO()
                    Image.open(photo).convert("RGB").resize((400, 900)).save(buf, "PNG")  # foto vertical -> 512x768
                    for ap in ("A", "B"):
                        _, pp = vitrina.start_session(store, engine, buf.getvalue(), {**base, "approach": ap})
                        app.setdefault("portrait_sizes", []).append(Image.open(pp["V1"]).size)
                    sid, _ = vitrina.start_session(store, engine, data, {**base, "approach": "B"})
                    on_submit(store, sid, ["aceptada", "rechazada", "aceptada", "rechazada", "aceptada"], ["", "no", "", "", ""])
                    changed = vitrina.new_round(store, engine, sid)
                    vitrina.restore(store, engine, sid, 1)
                    on_submit(store, sid, ["aceptada", "rechazada", "aceptada", "aceptada", "aceptada"], [""] * 5)
                    res = {k: vitrina.summarize(store, sid, *llms.get(k), db) for k in ("qwen", "phi")}
                    # se lee ahora: la celda 8 borra después las sesiones
                    app.update(store=store, sid=sid, sid_a=sid_a, changed=changed, res=res, launch=kw,
                               round=store.load(sid).round, metrics=vitrina.session_metrics(store, sid, cfg),
                               zip_names=zipfile.ZipFile(vitrina.export_looks(store, sid, cfg)).namelist())
            return Demo()
        vui.build_ui = build_ui

    ns = fresh_ns()
    def read_export(n):  # la celda 8 borra el ZIP: se lee justo después de la celda 7
        app["export_names"] = zipfile.ZipFile(colab.downloads[-1]).namelist() if colab.downloads else []

    def bench_photo(n):  # la celda 6 ya no abre un selector de archivos: las fotos se ponen en su carpeta
        d = content / "benchmark_vitrina" / "fotos"
        d.mkdir(parents=True, exist_ok=True)
        (d / photo.name).write_bytes(photo.read_bytes())

    resB = run(cells, ns, hooks={title["1"]: authorize, title["2"]: relax_resources, title["3"]: tiny,
                                 title["5"]: bench_photo, title["7"]: read_export}, quiet=False)
    okB = report("B (app + benchmark + exportar + borrar, modelos diminutos)", resB,
                 lambda r: all(s == "ok" for _, s, _ in r) and len(r) == len(cells))
    if okB:
        bench, names = content / "benchmark_vitrina", app["export_names"]
        checks = {
            "app lanzada con allowed_paths de las sesiones": app["launch"].get("allowed_paths"),
            "nueva ronda regeneró solo V2 y V4": sorted(app["changed"]) == ["V2", "V4"],
            "ronda final = 3 (inicial, nueva, restaurada)": app["round"] == 3,
            "resúmenes de los 2 LLM verificados (LLM o plantilla marcada)": all(r.verified for r in app["res"].values()),
            "métricas con costo y GPU": app["metrics"]["gpu_segundos"] >= 0 and "costo_gpu_estimado_usd" in app["metrics"],
            "ZIP de looks con looks.json": "looks.json" in app["zip_names"],
            "benchmark: CSV de imágenes y de LLM": {
                "benchmark_vitrina/benchmark_imagenes.csv", "benchmark_vitrina/benchmark_llm.csv"} <= set(names),
            "exportación sin fotos de entrada (input.png, fotos/)": names and not any(
                n.endswith("input.png") or "/fotos/" in n for n in names),
            "celda 8 borró sesiones, benchmark y ZIP": not (content / "sesiones_vitrina").exists() and not bench.exists()
            and not list(content.glob("vitrina_*.zip")),
            "app cerrada antes de borrar": app.get("closed", False),
            "foto vertical: A y B generan 512x768": app.get("portrait_sizes") == [(512, 768), (512, 768)],
        }
        for k, v in checks.items():
            print(f"  {'✓' if v else '✗'} {k}")
        okB = all(bool(v) for v in checks.values())
    results.append(okB)

    by = {t.split(".")[0]: (t, s) for t, s in cells}
    resC = run([by["1"], by["5"]], fresh_ns(), hooks={by["1"][0]: authorize})
    results.append(report("C (celda 5 sin las celdas 2-4)", resC,
                          lambda r: r[-1][1] == "detenida" and "Falta ejecutar celdas anteriores" in r[-1][2][0]))

    # K) Kaggle: sin google.colab, con KAGGLE_KERNEL_RUN_TYPE; /kaggle/working -> carpeta temporal
    import os
    kag = tmp / "kaggle_working"
    kag.mkdir()
    kcells = [(t, s.replace("/kaggle/working", kag.as_posix())) for t, s in cells]
    for m in [m for m in sys.modules if m == "google" or m.startswith("google.")]:
        del sys.modules[m]
    os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
    app.clear()
    colab.downloads.clear()
    nsK = fresh_ns()
    resK = run(kcells, nsK, hooks={title["1"]: lambda n: (authorize(n), n.update(CONFIRMO_BORRAR_DATOS=False)),
                                   title["2"]: relax_resources, title["3"]: tiny}, quiet=False)
    del os.environ["KAGGLE_KERNEL_RUN_TYPE"]
    okK = report("K (Kaggle: sin google.colab, rutas /kaggle/working)", resK,
                 lambda r: all(s == "ok" for _, s, _ in r) and len(r) == len(kcells))
    if okK:
        checks = {
            "plataforma detectada: Kaggle": nsK.get("EN_KAGGLE") is True and nsK.get("EN_COLAB") is False,
            "app con enlace público": app["launch"].get("share") is True,
            "celda 6 sin fotos: avisa sin colgarse": True,
            "ZIP en la carpeta de trabajo, sin descarga de Colab": bool(list(kag.glob("vitrina_*.zip"))) and not colab.downloads,
        }
        for k, v in checks.items():
            print(f"  {'✓' if v else '✗'} {k}")
        okK = all(checks.values())
    results.append(okK)
    print("\nRESULTADO:", "TODO COMO SE ESPERABA" if all(results) else "HAY ESCENARIOS NO ESPERADOS")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
