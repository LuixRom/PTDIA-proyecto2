"""Muestra qué zona protege el probador como «mano» en una foto (verde) sobre la máscara de manga larga (rojo).

Uso:  .venv\\Scripts\\python.exe tools\\debug_hands.py FOTO SALIDA.png [--set ruta=valor ...]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("foto")
    ap.add_argument("salida")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    import numpy as np
    import yaml
    from PIL import Image, ImageDraw

    from p2 import config, tryon, vitrina
    from p2.preprocess import prepare
    cfg = config.load("config/vitrina.yaml")
    for kv in a.set:
        path, val = kv.split("=", 1)
        *parents, last = path.split(".")
        node = cfg
        for k in parents:
            node = node[k]
        node[last] = yaml.safe_load(val)
    img, _ = prepare(Path(a.foto).read_bytes(), cfg["image_size"], fit="pad")
    seg, pose = tryon.Segmenter(cfg), tryon.PoseEstimator(cfg)
    engine = vitrina.Engine(cfg, seg, None, None, lambda im: False, None, pose=pose.detect)
    ctx = vitrina._context(engine, img, "B")
    lab = seg.labels(img)
    kps = pose.detect(img, tryon.person_box(lab, seg.id2label))
    b = cfg["approaches"]["B"]
    hp = b.get("hand_from_wrist", {})
    hands = tryon.hands_from_pose(lab, seg.id2label, kps, 0.3, hp.get("ahead", 0.35), hp.get("radius", 0.6)) & tryon.skin(img)
    m = np.asarray(ctx["masks"].get("superior_mangas", ctx["masks"]["superior"])) > 0
    over = np.asarray(img).copy()
    over[m] = (0.5 * over[m] + [115, 20, 20]).astype(np.uint8)
    over[hands] = (0.4 * over[hands] + [0, 150, 0]).astype(np.uint8)
    out = Image.fromarray(over)
    d = ImageDraw.Draw(out)
    for i in (7, 8, 9, 10):  # codos y muñecas
        x, y, s = kps[i]
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 0, 255), width=2)
    out.save(a.salida)
    print("muñecas:", kps[9].round(1).tolist(), kps[10].round(1).tolist(), "| píxeles mano:", int(hands.sum()))


if __name__ == "__main__":
    main()
