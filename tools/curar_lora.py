"""Copia las imágenes CURADAS del catálogo LoRA y arma el paquete para subir a Kaggle (fuera del repositorio).

Uso:  python tools/curar_lora.py piloto        (lee config/lora_curado_piloto.yaml)
Salida: C:/PTDIA/lora_data/<linea>_curado/<prenda>/  y  C:/PTDIA/lora_data/<nombre>_<linea>.zip con metadata.jsonl
        (formato «imagefolder» de Hugging Face: file_name + text).
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main(nombre="piloto"):
    cfg = yaml.safe_load((ROOT / "config/lora_catalogo.yaml").read_text(encoding="utf-8"))
    keep = yaml.safe_load((ROOT / f"config/lora_curado_{nombre}.yaml").read_text(encoding="utf-8"))
    linea = cfg[nombre]["linea"] if nombre in cfg else "mujer"
    src, dst = Path(cfg["salida"]) / linea, Path(cfg["salida"]) / f"{linea}_curado"
    shutil.rmtree(dst, ignore_errors=True)
    rows = []
    for pid, idx in keep.items():
        (dst / pid).mkdir(parents=True, exist_ok=True)
        for k in idx:
            f = src / pid / f"{k:02d}.png"
            if not f.exists():
                raise SystemExit(f"Falta {f}: genere el catálogo primero (tools/gen_catalogo_lora.py).")
            shutil.copy(f, dst / pid / f.name)
            rows.append({"file_name": f"{pid}_{f.name}", "text": f.with_suffix(".txt").read_text(encoding="utf-8"),
                         "src": f})
    z = Path(cfg["salida"]) / f"{nombre}_{linea}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in rows:
            zf.write(r.pop("src"), r["file_name"])
        zf.writestr("metadata.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print({p: len(i) for p, i in keep.items()}, "->", len(rows), "imágenes;", z, f"{z.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:])
