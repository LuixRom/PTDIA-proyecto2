"""Selección reproducible de candidatas SFHQ-T2I (semilla 2026). Nunca descarga imágenes.

Modo rápido (sin CSV): lista oficial de nombres de a_small_sample_new/ -> candidatas SDXL barajadas.
  python tools/select_images.py --kaggle-list --n 10
  (--kaggle-list consulta solo la API pública de Kaggle que lista NOMBRES de archivo; o use --sample-list FICHERO)
Modo completo (con el CSV oficial, si lo tiene): además exige model_used = prefijo y estilo fotográfico.
  python tools/select_images.py --kaggle-list --csv C:/PTDIA/sfhq_meta/SFHQ_T2I_dataset.csv --n 10
La revisión visual (1 cara, fotorrealista, frontal/3-4, sin oclusión) la hace una persona.
"""
import argparse
import csv
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PHOTO_PRESETS = {"photographic", "analog-film", "cinematic"}
NAME_RE = re.compile(r"^(.+)_image_\d{7}\.jpg$")
KAGGLE_LIST = ("https://www.kaggle.com/api/v1/datasets/list/"
               "selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models?pageSize=200")
SAMPLE_DIR = "a_small_sample_new/"


def candidates(rows, sample_names=None, models=("SDXL",), seed=2026):
    out = []
    for r in rows:
        fn, model = r["image_filename"], r["model_used"]
        prefix = NAME_RE.match(fn)
        # Doble comprobación: prefijo del nombre Y columna model_used deben coincidir.
        if not prefix or prefix.group(1) != model or model not in models:
            continue
        if sample_names is not None and fn not in sample_names:
            continue
        try:
            preset = json.loads(r.get("configs") or "{}").get("style_preset")
        except json.JSONDecodeError:
            preset = None
        if model == "SDXL" and preset not in PHOTO_PRESETS:
            continue
        out.append({"image_filename": fn, "model_used": model, "style_preset": preset, "text_prompt": r["text_prompt"]})
    out.sort(key=lambda x: x["image_filename"])
    random.Random(seed).shuffle(out)
    return out


def candidates_from_names(names, models=("SDXL",), seed=2026):
    """Sin CSV: modelo según el prefijo del nombre; estilo no verificado (se juzga a la vista)."""
    out = sorted({n for n in names if (m := NAME_RE.match(n)) and m.group(1) in models})
    random.Random(seed).shuffle(out)
    return [{"image_filename": n, "model_used": NAME_RE.match(n).group(1), "style_preset": None, "text_prompt": ""}
            for n in out]


def fetch_sample_names(opener=urllib.request.urlopen, max_pages=20):
    """Nombres de archivo de a_small_sample_new/ desde la API pública de Kaggle (solo metadatos)."""
    names, tok = [], ""
    for _ in range(max_pages):
        url = KAGGLE_LIST + (f"&pageToken={urllib.parse.quote(tok, safe='')}" if tok else "")
        with opener(url, timeout=30) as resp:
            d = json.load(resp)
        names += [f["name"][len(SAMPLE_DIR):] for f in d.get("datasetFiles", []) if f["name"].startswith(SAMPLE_DIR)]
        tok = d.get("nextPageTokenNullable") or ""
        if not tok:
            break
    return names


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # consolas Windows cp1252: no fallar al imprimir «→»
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kaggle-list", action="store_true", help="obtener los nombres de a_small_sample_new/ de Kaggle")
    src.add_argument("--sample-list", help="fichero con nombres de a_small_sample_new/ (uno por línea)")
    ap.add_argument("--csv", help="SFHQ_T2I_dataset.csv oficial (opcional)")
    ap.add_argument("--save-list", help="guardar la lista de nombres obtenida para reutilizarla")
    ap.add_argument("--models", default="SDXL")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)
    if a.kaggle_list:
        names = fetch_sample_names()
    else:
        names = [l.strip().split("/")[-1] for l in open(a.sample_list, encoding="utf-8") if l.strip()]
    if a.save_list:
        Path(a.save_list).write_text("\n".join(names) + "\n", encoding="utf-8")
    models = tuple(a.models.split(","))
    if a.csv:
        with open(a.csv, newline="", encoding="utf-8") as f:
            cands = candidates(list(csv.DictReader(f)), set(names), models)
    else:
        cands = candidates_from_names(names, models)
    print(f"{len(names)} nombres en {SAMPLE_DIR}; {len(cands)} candidatas {models}"
          f"{'' if a.csv else ' (sin CSV: estilo no verificado)'}. Primeras {a.n}:")
    for i, c in enumerate(cands[: a.n], start=1):
        print(f"{i:>2}. {c['image_filename']}\t{c['style_preset'] or ''}\t{c['text_prompt'][:100]}")
    if cands:
        print(f"\nDescargue SOLO la que elija desde Kaggle: {SAMPLE_DIR}<nombre> → data/sfhq_t2i/")
    return cands


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
