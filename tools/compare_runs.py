"""Hoja comparativa: misma foto y mismos looks generados en varias corridas de tools/eval_local.py (una fila cada una).

Uso:  python tools/compare_runs.py SALIDA.png CARPETA_CORRIDA1 CARPETA_CORRIDA2 ... [--foto chica-photo] [--box x0,y0,x1,y1]
Cada carpeta debe tener sesiones/<id>/ (se toma la sesión cuya foto de entrada coincide en tamaño con la primera).
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("salida")
    ap.add_argument("corridas", nargs="+")
    ap.add_argument("--box", default="")
    ap.add_argument("--sesion", type=int, default=0, help="índice de sesión (orden alfabético) en cada corrida")
    a = ap.parse_args()
    rows, labels = [], []
    for c in a.corridas:
        d = sorted(Path(c, "sesiones").iterdir())[a.sesion]
        s = json.loads((d / "decisions.json").read_text(encoding="utf-8"))
        im0 = Image.open(d / "input.png").convert("RGB")
        box = tuple(int(v) for v in a.box.split(",")) if a.box else (0, 0, *im0.size)
        rows.append([im0.crop(box)] + [Image.open(d / v["image_file"]).convert("RGB").crop(box) for v in s["variations"]])
        labels.append(Path(c).name)
    w, h = rows[0][0].size
    out = Image.new("RGB", (len(rows[0]) * (w + 6) + 110, len(rows) * (h + 6)), "white")
    dr = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        dr.text((4, r * (h + 6) + h // 2), labels[r], fill=(0, 0, 0))
        for k, im in enumerate(row):
            out.paste(im, (110 + k * (w + 6), r * (h + 6)))
    out.save(a.salida)


if __name__ == "__main__":
    main()
