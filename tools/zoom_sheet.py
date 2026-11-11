"""Amplía la zona del torso de cada look de una hoja de tools/eval_local.py para revisarla de cerca.

Uso:  python tools/zoom_sheet.py HOJA.png SALIDA.png [--y0 0.05 --y1 0.6 --cols 7]
"""
import argparse

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hoja")
    ap.add_argument("salida")
    ap.add_argument("--y0", type=float, default=0.05)
    ap.add_argument("--y1", type=float, default=0.62)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--alto", type=int, default=520)
    a = ap.parse_args()
    s = Image.open(a.hoja).convert("RGB")
    tw = s.width // a.cols
    h = s.height - 22
    tiles = []
    for k in range(a.cols):
        t = s.crop((k * tw, 22 + int(h * a.y0), k * tw + tw - 8, 22 + int(h * a.y1)))
        tiles.append(t.resize((int(t.width * a.alto / t.height), a.alto)))
    out = Image.new("RGB", (sum(t.width for t in tiles) + 6 * len(tiles), a.alto), "white")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + 6
    out.save(a.salida)


if __name__ == "__main__":
    main()
