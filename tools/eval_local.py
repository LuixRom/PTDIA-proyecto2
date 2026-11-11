"""Evaluación visual del probador con los modelos REALES en una GPU local (p. ej. 4 GB, con descarga a CPU).

Uso:  .venv\\Scripts\\python.exe tools\\eval_local.py CARPETA_FOTOS CARPETA_SALIDA [--ocasion trabajo]
      [--hombre "foto1.jpg,foto2.jpg"] [--candidatos 1]

Por cada foto genera los 5 looks (modo prueba: semillas fijas) con el mismo código que la app y guarda una hoja:
original | máscara (rojo) + manos protegidas (verde) + esqueleto | V1..V5. Las fotos no se copian al repositorio.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # usa los modelos ya descargados (tools/download_models.py), sin red

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sheet(d, s, name, out):
    inp = Image.open(d / "input.png").convert("RGB")
    key = sorted(p.stem for p in d.glob("mask_*.png"))
    mask = Image.open(d / f"{'mask_superior_mangas' if 'mask_superior_mangas' in key else 'mask_superior'}.png")
    dbg = Image.composite(Image.new("RGB", inp.size, (230, 40, 40)), inp, mask.point(lambda v: 110 if v else 0))
    if (d / "control.png").exists():
        ctrl = Image.open(d / "control.png").convert("RGB")
        dbg = Image.composite(ctrl, dbg, ctrl.convert("L").point(lambda v: 255 if v > 10 else 0))
    tiles = [("original", inp), ("máscara + pose", dbg)] + [
        (f"{v.id} {v.attribute[:28]}", Image.open(d / v.image_file).convert("RGB")) for v in s.variations]
    w, h = inp.size
    canvas = Image.new("RGB", (len(tiles) * (w + 8), h + 22), "white")
    dr = ImageDraw.Draw(canvas)
    for k, (cap, im) in enumerate(tiles):
        canvas.paste(im.resize((w, h)), (k * (w + 8), 22))
        dr.text((k * (w + 8) + 4, 4), cap, fill=(0, 0, 0))
    canvas.save(out / f"hoja_{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fotos")
    ap.add_argument("salida")
    ap.add_argument("--ocasion", default="trabajo")
    ap.add_argument("--hombre", default="", help="fotos (separadas por coma) de la línea hombre")
    ap.add_argument("--candidatos", type=int, default=1)
    ap.add_argument("--lora", default="", help="ruta del .safetensors del LoRA «Estilo de la tienda» (nivel avanzado)")
    ap.add_argument("--lora-escala", type=float, default=1.0)
    ap.add_argument("--api", action="store_true", help="enfoque API (OpenAI, de pago, envía la foto): usa la clave del .env")
    ap.add_argument("--solo", default="", help="procesar solo estas fotos (coma)")
    ap.add_argument("--set", action="append", default=[], help="cambia la config: ruta.con.puntos=valor_yaml")
    a = ap.parse_args()

    from p2 import config, tryon, vitrina
    from p2.decisions import DecisionStore
    cfg = config.load("config/vitrina.yaml")
    cfg["approaches"]["B"]["candidates"] = a.candidatos
    import yaml
    for kv in a.set:  # experimentos sin tocar config/vitrina.yaml
        path, val = kv.split("=", 1)
        *parents, last = path.split(".")
        node = cfg
        for k in parents:
            node = node[k]
        node[last] = yaml.safe_load(val)
        print("config:", path, "=", node[last])
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    seg, pose = tryon.Segmenter(cfg), tryon.PoseEstimator(cfg)
    pipe = tryon.load_inpaint(cfg, offload=True)
    clip = None
    if a.candidatos > 1:
        from p2 import metrics
        clip = metrics.Models(cfg, use_lpips=False)
    print(f"Modelos cargados en {time.perf_counter() - t0:.0f} s")
    engine = vitrina.Engine(cfg, seg, None, lambda img, mask, spec, control=None: tryon.gen_b(pipe, img, mask, spec, cfg, control),
                            lambda img: tryon.nsfw_input(pipe, img), clip, pose=pose.detect)
    if a.lora:
        from p2.lora import LoraSwitch
        engine.lora = LoraSwitch(pipe.unet, a.lora, a.lora_escala)
        print("LoRA:", a.lora, "| capas:", len(engine.lora.pairs), "| escala", a.lora_escala)
    if a.api:
        from p2 import api_images
        engine.api = api_images.ApiEditor(**cfg["approaches"]["API"]["api"])
        assert engine.api.available(), "No hay clave de la API en el .env"
    store = DecisionStore(out / "sesiones")
    men = {x.strip() for x in a.hombre.split(",") if x.strip()}
    only = {x.strip() for x in a.solo.split(",") if x.strip()}
    for f in sorted(Path(a.fotos).iterdir()):
        if not f.is_file() or (only and f.name not in only):
            continue
        t1 = time.perf_counter()
        try:
            sid, _ = vitrina.start_session(store, engine, f.read_bytes(), dict(
                mode="ocasion", occasion=a.ocasion, linea="hombre" if f.name in men else "mujer", approach="API" if a.api else "B",
                api_consent=a.api,
                seed_mode="prueba", own_photo=True, adult=True, llm="qwen", lora=bool(a.lora)))
        except Exception as e:  # noqa: BLE001  (guardrail o fallo: se informa y se sigue con la siguiente)
            print(f"{f.name}: {type(e).__name__}: {e}")
            continue
        s = store.load(sid)
        sheet(store.root / sid, s, f.stem.replace(" ", "_"), out)
        print(f"{f.name}: {time.perf_counter() - t1:.0f} s | manos ubicadas {s.options.get('hands_detected')} | "
              f"alertas filtro {sum(v.metrics.get('nsfw_alertas', 0) for v in s.variations)}")


if __name__ == "__main__":
    main()
