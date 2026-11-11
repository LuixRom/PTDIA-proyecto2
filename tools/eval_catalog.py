"""Prueba cada prenda del catálogo sobre varias fotos (modo catálogo, semillas fijas) para depurar el catálogo.

Uso:  .venv\\Scripts\\python.exe tools\\eval_catalog.py CARPETA_FOTOS SALIDA --mujer "a.jpg,b.jpg" --hombre "c.jpg,d.jpg"
Por foto genera grupos de 5 prendas de su línea y guarda hojas «catalogo_<foto>_<n>.png» con el nombre de cada prenda.
"""
import argparse
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fotos")
    ap.add_argument("salida")
    ap.add_argument("--mujer", default="")
    ap.add_argument("--hombre", default="")
    ap.add_argument("--solo", default="", help="IDs de prendas a probar (coma); vacío = todo el catálogo")
    a = ap.parse_args()
    from p2 import config, tryon, vitrina
    from p2.decisions import DecisionStore
    cfg = config.load("config/vitrina.yaml")
    cfg["approaches"]["B"]["candidates"] = 1
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    seg, pose = tryon.Segmenter(cfg), tryon.PoseEstimator(cfg)
    pipe = tryon.load_inpaint(cfg, offload=True)
    engine = vitrina.Engine(cfg, seg, None, lambda img, mask, spec, control=None: tryon.gen_b(pipe, img, mask, spec, cfg, control),
                            lambda img: False, None, pose=pose.detect)
    store = DecisionStore(out / "sesiones")
    only = {x for x in a.solo.split(",") if x}
    for linea, fotos in (("mujer", a.mujer), ("hombre", a.hombre)):
        items = [k for k in vitrina.catalog_for(cfg, linea) if not only or k in only]
        for foto in [f for f in fotos.split(",") if f]:
            data = (Path(a.fotos) / foto).read_bytes()
            for n in range(0, len(items), 5):
                chunk = items[n:n + 5]
                t0 = time.perf_counter()
                sid, paths = vitrina.start_session(store, engine, data, dict(
                    mode="catalogo", items=chunk, linea=linea, approach="B", seed_mode="prueba",
                    own_photo=True, adult=True, llm="qwen"))
                d = store.root / sid
                tiles = [("original", Image.open(d / "input.png").convert("RGB"))]
                tiles += [(chunk[k % len(chunk)], Image.open(paths[f"V{k + 1}"]).convert("RGB")) for k in range(len(chunk))]
                w, h = tiles[0][1].size
                sheet = Image.new("RGB", (len(tiles) * (w + 6), h + 20), "white")
                dr = ImageDraw.Draw(sheet)
                for k, (cap, im) in enumerate(tiles):
                    sheet.paste(im, (k * (w + 6), 20))
                    dr.text((k * (w + 6) + 4, 4), cap, fill=(0, 0, 0))
                name = f"catalogo_{Path(foto).stem.replace(' ', '_')}_{n // 5 + 1}.png"
                sheet.save(out / name)
                print(f"{name}: {', '.join(chunk)} ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
