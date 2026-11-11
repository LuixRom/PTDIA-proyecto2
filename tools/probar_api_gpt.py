"""Prueba EXTRA (no forma parte del sistema): los mismos 5 looks hechos con la API de imágenes de OpenAI.

Sirve para el informe: «¿y con una API comercial cómo sale?». Cuesta dinero por imagen y ENVÍA LA FOTO a un tercero:
por eso no envía nada sin --confirmo. La clave se lee de la variable de entorno OPENAI_API_KEY o del archivo .env
(nombres aceptados: OPENAI_API_KEY o API-KEY) y NUNCA se imprime ni se guarda. El .env está en .gitignore.

Uso:  .venv\\Scripts\\python.exe tools\\probar_api_gpt.py --confirmo [--foto RUTA] [--ocasion trabajo] [--max 5]
"""
import argparse
import base64
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from p2.api_images import KEY_NAMES, cargar_clave, editar, multipart, ocultar, prompt_look  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foto", default="C:/PTDIA/imagenes_prueba_probador/mujer.png")
    ap.add_argument("--ocasion", default="trabajo")
    ap.add_argument("--linea", default="mujer")
    ap.add_argument("--max", type=int, default=5, help="máximo de imágenes (cada una cuesta)")
    ap.add_argument("--modelo", default="gpt-image-2")
    ap.add_argument("--size", default="1024x1536")
    ap.add_argument("--salida", default="C:/PTDIA/eval_probador/api_gpt")
    ap.add_argument("--confirmo", action="store_true",
                    help="confirmo que acepto enviar esta foto a OpenAI y que cada imagen tiene costo")
    a = ap.parse_args()

    clave = cargar_clave()
    if not clave:
        raise SystemExit("No encuentro la clave (variable OPENAI_API_KEY o archivo .env con OPENAI_API_KEY=...).")
    if not a.confirmo:
        raise SystemExit(f"Esto ENVÍA {a.foto} a OpenAI y genera hasta {a.max} imágenes con costo. "
                         "Repita con --confirmo si está de acuerdo.")

    from PIL import Image
    from p2 import config, guardrails, vitrina
    cfg = config.load("config/vitrina.yaml")
    looks, _ = vitrina.looks_for(cfg, dict(mode="ocasion", occasion=a.ocasion, linea=a.linea, seed_mode="prueba"))
    img = Image.open(a.foto).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for k, lk in enumerate(looks[:a.max], start=1):
        t0 = time.perf_counter()
        try:
            r = editar(clave, a.modelo, buf.getvalue(), prompt_look(lk["text"]), a.size)
            d = r["data"][0]
            datos = base64.b64decode(d["b64_json"]) if d.get("b64_json") else urllib.request.urlopen(d["url"]).read()
            res = Image.open(io.BytesIO(datos)).convert("RGB")
            guardrails.label(res, cfg["guardrails"]["label"]).save(out / f"V{k}.png")
            log.append({"id": f"V{k}", "look": lk["name"], "seg": round(time.perf_counter() - t0, 1),
                        "uso": r.get("usage"), "ok": True})
            print(f"V{k} {lk['name']}: OK en {log[-1]['seg']} s | uso: {r.get('usage')}", flush=True)
        except Exception as e:  # noqa: BLE001
            log.append({"id": f"V{k}", "look": lk["name"], "ok": False, "error": ocultar(e)})
            print(f"V{k} {lk['name']}: FALLÓ — {ocultar(e)}", flush=True)
    img.save(out / "entrada.png")
    (out / "registro.json").write_text(json.dumps({"modelo": a.modelo, "size": a.size, "foto": Path(a.foto).name,
                                                   "looks": log}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
