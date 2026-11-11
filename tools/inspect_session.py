"""Vista ampliada de una sesión de tools/eval_local.py: entrada, máscaras y looks, lado a lado.

Uso:  python tools/inspect_session.py CARPETA_SESION SALIDA.png [--ids V1,V4] [--y0 0 --y1 0.6]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sesion")
    ap.add_argument("salida")
    ap.add_argument("--ids", default="V1,V4")
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--y1", type=float, default=0.6)
    a = ap.parse_args()
    d = Path(a.sesion)
    inp = Image.open(d / "input.png").convert("RGB")
    tiles = [("entrada", inp)]
    for m in sorted(d.glob("mask_*.png")):
        mk = Image.open(m).convert("L")
        tiles.append((m.stem, Image.composite(Image.new("RGB", inp.size, (230, 40, 40)), inp, mk.point(lambda v: 140 if v else 0))))
    for vid in a.ids.split(","):
        f = sorted(d.glob(f"{vid}_r*.png"))[-1]
        tiles.append((f.stem, Image.open(f).convert("RGB")))
    w, h = inp.size
    box = (0, int(h * a.y0), w, int(h * a.y1))
    crops = [(c, t.crop(box)) for c, t in tiles]
    cw, ch = crops[0][1].size
    out = Image.new("RGB", (len(crops) * (cw + 6), ch + 18), "white")
    dr = ImageDraw.Draw(out)
    for k, (c, t) in enumerate(crops):
        out.paste(t, (k * (cw + 6), 18))
        dr.text((k * (cw + 6) + 3, 3), c, fill=(0, 0, 0))
    out.save(a.salida)


if __name__ == "__main__":
    main()
