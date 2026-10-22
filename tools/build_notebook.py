"""Genera run_colab.ipynb (nbformat 4). Uso: python tools/build_notebook.py

El notebook lleva el código del proyecto INCLUIDO (celda 1b: ZIP en base64 + SHA-256), así que en Colab
basta con subir este .ipynb: no hace falta subir un ZIP aparte ni pueden mezclarse versiones.
La simulación completa en local está en tools/simulate_colab.py.
"""
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD, CODE = "markdown", "code"

CELLS = [
(MD, r"""# Proyecto 2 — Piloto A1 vs A2 (SD 1.5 img2img) en Google Colab

**Cómo usarlo:** *Entorno de ejecución › Cambiar tipo de entorno de ejecución › **GPU T4***. Después ejecute las celdas **en orden, de arriba abajo** (o *Entorno de ejecución › Ejecutar todas*).
- **Sin autorizaciones** (por defecto), las celdas 1–5 comprueban todo sin descargar nada y las 6–8 muestran «⏸ omitida». No aparece ningún error.
- **Para el piloto**, marque en la celda 1 `AUTORIZO_DESCARGAR_MODELOS` y `AUTORIZO_EJECUTAR_PILOTO` y vuelva a ejecutar todas las celdas.
- **No hay que subir ningún ZIP**: el código y la imagen de entrada van incluidos en este notebook (celda 1b).
- **App final (celda 9):** marque `AUTORIZO_DESCARGAR_MODELOS` y `AUTORIZO_APP`. La app aparece dentro de la celda (sin enlace público): subir imagen → 5 variaciones (A2, strength 0.55) → aceptar/rechazar cada una → resumen redactado por el LLM Qwen2.5-1.5B y comprobado por el verificador. La celda 10 descarga las sesiones registradas.

**Qué hace:** compara la línea base **A1** (un prompt genérico, 5 semillas; config. de `Untitled1.ipynb` celda 44) con **A2** (5 prompts de atributo, mismas semillas) sobre **1 imagen** de entrada.
- **Piloto 2:** se repite para cada strength de `generation.strengths` (0.55 y 0.70): 10 imágenes + 1 repetición de determinismo por strength, más 1 calentamiento (23 en total). El piloto 1 (strength 0.40) ya se ejecutó.
- Exporta imágenes, `runs.jsonl` (semillas y parámetros), métricas, conjunto ciego y versiones del entorno en un ZIP que se descarga.

**Descargas (solo con autorización):** SD 1.5 fp16 ≈ 2.75 GB + CLIP ViT-B/32 ≈ 0.61 GB → **≈ 3.4 GB** (estimación). **No instala nada:** usa los paquetes de Colab y valida su compatibilidad sin red. Sin LPIPS (no está instalado en Colab): se registra como métrica no disponible.

**Licencias:**
- SD 1.5: CreativeML Open RAIL-M.
- CLIP: solo evaluación, nunca para identificar personas.
- Imagen de entrada: SFHQ-T2I (sintética, generada con SDXL; licencia MIT según la ficha de Kaggle, no es una verificación legal).

**Resultados exploratorios:** n = 1 imagen; no son evidencia estadística generalizable."""),

(CODE, r"""#@title 1. Parámetros y autorizaciones
#@markdown Las dos autorizaciones de descarga y ejecución están desactivadas por defecto.
AUTORIZO_DESCARGAR_MODELOS = False  #@param {type:"boolean"}
AUTORIZO_EJECUTAR_PILOTO = False  #@param {type:"boolean"}
#@markdown Continuar sin LPIPS (decisión del proyecto: lpips no está instalado en Colab y no se instala).
ACEPTO_SIN_LPIPS = True  #@param {type:"boolean"}
#@markdown App final (celda 9): usa SD 1.5 y descarga además el LLM Qwen2.5-1.5B (~3.1 GB). Requiere también AUTORIZO_DESCARGAR_MODELOS.
AUTORIZO_APP = False  #@param {type:"boolean"}
#@markdown Prueba con compañeros: crea un enlace público temporal *.gradio.live (dura mientras este Colab siga abierto, máx. 72 h). Cualquiera con el enlace puede usar la app.
ENLACE_PUBLICO = False  #@param {type:"boolean"}
#@markdown Al terminar la prueba (y tras descargar el ZIP de la celda 10): borra sesiones, imágenes y la BD temporal (celda 11).
CONFIRMO_BORRAR_DATOS = False  #@param {type:"boolean"}
EXPORTAR_A_DRIVE = False  #@param {type:"boolean"}
#@markdown Origen del código: "incluido" (recomendado) o un ZIP subido/en Drive.
FUENTE_CODIGO = "incluido"  #@param ["incluido", "subir", "drive"]
RUTA_ZIP_EN_DRIVE = "MyDrive/proyecto2/proyecto2_colab.zip"  #@param {type:"string"}
WORKDIR = "/content/p2run"
VALIDATION_PATH = "env/validation.json"
EXPECTED_P2_VERSION = "__P2_VERSION__"  # fijado al generar el notebook


def requiere(*nombres):
    # Detiene la celda con una explicación si no se ejecutaron antes las celdas previas.
    faltan = [n for n in nombres if n not in globals()]
    if faltan:
        raise SystemExit(f"Falta ejecutar celdas anteriores (no existe: {', '.join(faltan)}). "
                         "Use «Entorno de ejecución › Ejecutar todas» o ejecute las celdas desde la 1 en orden.")


PILOTO_AUTORIZADO = bool(AUTORIZO_DESCARGAR_MODELOS and AUTORIZO_EJECUTAR_PILOTO)
APP_AUTORIZADA = bool(AUTORIZO_DESCARGAR_MODELOS and AUTORIZO_APP)
print("Autorizaciones:", dict(descargar=AUTORIZO_DESCARGAR_MODELOS, piloto=AUTORIZO_EJECUTAR_PILOTO, app=AUTORIZO_APP,
                              sin_lpips=ACEPTO_SIN_LPIPS, drive=EXPORTAR_A_DRIVE, codigo=FUENTE_CODIGO))"""),

(CODE, r"""#@title 1b. Código del proyecto incluido (no editar)
PAYLOAD_SHA256 = "__PAYLOAD_SHA256__"
PAYLOAD_B64 = "__PAYLOAD_B64__"
print(f"Código incluido: {len(PAYLOAD_B64) * 3 // 4 / 1e3:.0f} kB, sha256 {PAYLOAD_SHA256[:12]}…")"""),

(CODE, r"""#@title 2. Preparar el código del proyecto y verificar su integridad
import base64, hashlib, os, shutil, sys, zipfile
from pathlib import Path

try:
    import google.colab  # noqa: F401
except ImportError:
    raise SystemExit("Este notebook está pensado para Google Colab. En Windows use los tests locales (README.md).")

if FUENTE_CODIGO == "incluido":
    requiere("PAYLOAD_B64", "PAYLOAD_SHA256")
    data = base64.b64decode(PAYLOAD_B64)
    if hashlib.sha256(data).hexdigest() != PAYLOAD_SHA256:
        raise SystemExit("El código incluido está dañado. Vuelva a descargar run_colab.ipynb del proyecto.")
    zip_path = Path("/content/p2_codigo_incluido.zip")
    zip_path.write_bytes(data)
elif FUENTE_CODIGO == "subir":
    from google.colab import files
    subidos = files.upload()
    if not subidos:
        raise SystemExit("No se subió ningún archivo. Suba proyecto2_colab.zip.")
    zip_path = Path("/content") / next(iter(subidos))
else:
    from google.colab import drive
    drive.mount("/content/drive")
    zip_path = Path("/content/drive") / RUTA_ZIP_EN_DRIVE
    if not zip_path.exists():
        raise SystemExit(f"No existe {zip_path}. Revise RUTA_ZIP_EN_DRIVE.")

print("Código:", zip_path.name, "| SHA-256:", hashlib.sha256(zip_path.read_bytes()).hexdigest())
os.chdir(Path(WORKDIR).parent)  # no borrar la carpeta en la que estamos (al repetir esta celda)
if Path(WORKDIR).exists():
    shutil.rmtree(WORKDIR)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(WORKDIR)

bad = []
for line in (Path(WORKDIR) / "MANIFEST.sha256").read_text().splitlines():
    if not line.strip():
        continue
    digest, rel = line.split("  ", 1)
    p = Path(WORKDIR) / rel
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        bad.append(rel)
if bad:
    raise SystemExit(f"El código no coincide con su manifiesto ({len(bad)} archivos): {bad[:5]}.")
os.chdir(WORKDIR)
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)
# Descarta módulos del proyecto ya cargados en este kernel (de una ejecución anterior).
for mod in [m for m in sys.modules if m.split(".")[0] in ("p2", "experiments", "tools")]:
    del sys.modules[mod]
import p2
if getattr(p2, "__version__", None) != EXPECTED_P2_VERSION:
    raise SystemExit(f"El código contiene p2 {getattr(p2, '__version__', '(sin versión: ZIP antiguo)')}, "
                     f"pero este notebook espera {EXPECTED_P2_VERSION}. Use FUENTE_CODIGO = \"incluido\".")
CODIGO_OK = True
print("Código verificado en", WORKDIR, "| p2", p2.__version__)"""),

(CODE, r"""#@title 3. Comprobar GPU, VRAM, RAM, disco y versiones (no instala nada)
requiere("CODIGO_OK")
import json
from p2 import config, env_check

cfg = config.load("config/pilot.yaml")
info = env_check.collect("/content")
print(json.dumps({k: v for k, v in info.items() if k != "packages"}, indent=2))
print("Paquetes:", json.dumps({k: v for k, v in info["packages"].items() if v}, indent=2))
probs = env_check.problems(info, cfg["resources"]["t4"])
if probs:
    raise SystemExit("No se puede continuar:\n- " + "\n- ".join(probs))
print("Recursos suficientes. Config sha256:", cfg["_sha256"][:12])"""),

(CODE, r"""#@title 4. Validación local de compatibilidad (sin red, sin pip, no instala nada)
requiere("cfg", "info")
from importlib import metadata
from pathlib import Path

Path(VALIDATION_PATH).unlink(missing_ok=True)  # una validación anterior nunca se reutiliza
state = env_check.validate(cfg)
state = env_check.import_smoke(state)  # importa de verdad diffusers/transformers (sin red ni descargas)
for m in state["import_smoke"]["messages"]:
    print("Importación:", m)
Path("env").mkdir(exist_ok=True)
Path("env/installed_packages.txt").write_text(
    "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())) + "\n")
print("Versiones:", {k: v for k, v in state["versions"].items() if v})
print(f"Comprobaciones correctas: {len(state['checked'])}")
for k, o in state["optional"].items():
    print(f"Opcional {k}: {'disponible ' + o['installed'] if o['available'] else 'NO disponible'} {o['problems'] or ''}")
for p in state["problems"]:
    print(" ✗", p)
for u in state["uncertain"]:
    print(" ?", u)
env_check.save_validation(state, VALIDATION_PATH)
if not state["ok"]:
    raise SystemExit("Validación NO superada: hay incompatibilidades o dependencias no verificables (✗/?). "
                     "No se instala ni se cambia nada automáticamente; las celdas 6 y 7 quedarán bloqueadas.")
print("Validación correcta →", VALIDATION_PATH)"""),

(CODE, r"""#@title 5. Comprobar la imagen de entrada (manifiesto y licencia)
requiere("cfg")
from p2 import dataset

data_dir = Path(cfg["dataset"]["dir"])
name = cfg["dataset"]["pilot_image"]
if not name:
    raise SystemExit("config/pilot.yaml: dataset.pilot_image está vacío. Registre la imagen con "
                     "tools/register_image.py --pilot (README, «Imagen de entrada») y regenere el notebook.")
probs = dataset.check_dataset_dir(data_dir, cfg["dataset"]["allowed_models"], name)
if probs:
    raise SystemExit("Imagen de entrada no válida:\n- " + "\n- ".join(probs))
row = {r["image_filename"]: r for r in dataset.read_manifest(data_dir / "MANIFEST.csv")[0]}[name]
input_path = data_dir / name
print("Imagen de entrada OK:", name, "| modelo:", row["model_used"], "| estilo:", row["style_preset"])
print("Fuente:", row["source"])
print("Licencia:", row["license"])"""),

(CODE, r"""#@title 6. Descargar SD 1.5 (revisión fijada) y generar A1 y A2 para cada strength (reanudable)
requiere("cfg", "input_path")
try:
    validation = env_check.require_validation(cfg, VALIDATION_PATH)
except env_check.ValidationError as e:
    raise SystemExit(f"Bloqueado: {e}")
import torch
from experiments import pilot
from p2 import generate

torch.backends.cudnn.benchmark = False
OUT = Path("/content/results") / f"{cfg['run_name']}_{cfg['_sha256'][:8]}"
STRENGTHS = cfg["generation"]["strengths"]
pipe = generate.load_sd15(cfg)  # se carga una vez y se reutiliza para cada strength
for i, s in enumerate(STRENGTHS):
    res = pilot.generate_all(cfg, input_path, OUT / f"strength_{s:.2f}", backend="sd15", validation=validation,
                             strength=s, pipe=pipe, warmup=(i == 0))
    print(f"strength {s}: {res['n_main']} imágenes | determinismo (misma semilla -> mismo archivo): {res['determinism_identical']}")
del pipe"""),

(CODE, r"""#@title 7. Liberar SD y calcular métricas (SSIM, CLIP, LPIPS si está disponible, caras, NSFW)
requiere("cfg", "OUT", "STRENGTHS")
try:
    validation = env_check.require_validation(cfg, VALIDATION_PATH)
except env_check.ValidationError as e:
    raise SystemExit(f"Bloqueado: {e}")
use_lpips = validation["optional"]["lpips"]["available"]
if not use_lpips:
    if not ACEPTO_SIN_LPIPS:
        raise SystemExit("LPIPS no está instalado (dependencia opcional). Marque ACEPTO_SIN_LPIPS en la celda 1 "
                         "para continuar solo con SSIM y CLIP.")
    print("⚠ Se continúa SIN LPIPS (aceptado). Quedará registrado en set_metrics.csv y LIMITATIONS.md.")
import gc, torch
from experiments import pilot
from p2 import metrics

gc.collect(); torch.cuda.empty_cache()
models = metrics.Models(cfg, use_lpips=use_lpips)
for s in STRENGTHS:
    rows, sets = pilot.compute_metrics(cfg, OUT / f"strength_{s:.2f}", models, validation=validation)
summary = pilot.sweep_summary(OUT, STRENGTHS)
print("--- sweep_summary.csv (CLIPΔ > 0: el atributo se aplicó; clip_img_sim: parecido con el original)")
print((OUT / "sweep_summary.csv").read_text(encoding="utf-8"))
print("Exploratorio: 1 imagen x 5 variaciones por método y strength. No generalizar.")"""),

(CODE, r"""#@title 8. Hojas de contacto, conjunto ciego, manifiesto y descarga de resultados
requiere("cfg", "OUT", "STRENGTHS", "summary")
import shutil
from datetime import datetime
from experiments import pilot
from p2 import export

for s in STRENGTHS:
    pilot.contact_sheets(OUT / f"strength_{s:.2f}")
shutil.copy("env/installed_packages.txt", OUT / "installed_packages.txt")
n = pilot.finalize(cfg, OUT, "config/pilot.yaml", VALIDATION_PATH)
zip_out = Path("/content") / f"{OUT.name}_{datetime.now():%Y%m%d-%H%M}.zip"
export.zip_dir(OUT, zip_out)
print(f"{n} archivos en el manifiesto -> {zip_out} ({zip_out.stat().st_size/1e6:.1f} MB)")
if EXPORTAR_A_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive/proyecto2/resultados"); dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(zip_out, dest / zip_out.name); print("Copiado a", dest)
from google.colab import files
files.download(str(zip_out))"""),

(CODE, r"""#@title 9. App final: subir imagen → 5 variaciones (A2, strength 0.55) → aceptar/rechazar → resumen del LLM
requiere("cfg")
try:
    validation = env_check.require_validation(cfg, VALIDATION_PATH)
except env_check.ValidationError as e:
    raise SystemExit(f"Bloqueado: {e}")
try:
    import gradio  # noqa: F401  (preinstalado en Colab)
except ImportError:
    raise SystemExit("gradio no está instalado en este entorno; este notebook no instala paquetes.")
import gc, torch
from p2 import generate, llm as p2llm, ui
from p2.decisions import DecisionStore

for _v in ("pipe", "models"):  # libera modelos de las celdas 6-7 si siguen en memoria
    globals().pop(_v, None)
gc.collect(); torch.cuda.empty_cache()
APP_GEN = {**cfg["generation"], "strength": cfg["app"]["strength"]}
app_pipe = generate.load_sd15(cfg)
generate_one = lambda im, spec: generate.sd15_generate(app_pipe, im, spec, APP_GEN)  # noqa: E731
print("Cargando el LLM explicador:", cfg["llm"]["repo"], "(la primera vez descarga ~3.1 GB)")
llm_fn, llm_name = p2llm.load_llm(cfg)
from p2.tempdb import TempDB

store = DecisionStore(Path("/content") / cfg["app"]["sessions_dir"])
db = TempDB(store.root / "agente_tmp.sqlite")  # BD temporal: respuestas del agente + SUS (se borra en la celda 11)
demo = ui.build_ui(store, cfg, method=cfg["app"]["method"], generate_one=generate_one, out_dir=store.root,
                   llm=llm_fn, llm_name=llm_name, gen=APP_GEN, db=db)
print(f"App lista: método {cfg['app']['method']}, strength {APP_GEN['strength']}, LLM {llm_name}. "
      f"Sesiones y BD temporal en {store.root}.")
# allowed_paths: gradio solo sirve archivos de carpetas autorizadas; las sesiones están fuera de la carpeta de trabajo.
demo.launch(share=ENLACE_PUBLICO, inline=not ENLACE_PUBLICO, height=1100, quiet=True, show_error=True,
            allowed_paths=[str(store.root)])
if ENLACE_PUBLICO:
    print("ENLACE PÚBLICO TEMPORAL (compártalo con los participantes):", demo.share_url)
    print("Funciona mientras este Colab siga abierto. Al terminar: celda 10 (descargar) y celda 11 (borrar).")
else:
    print("App dentro de esta celda, sin enlace público.")"""),

(CODE, r"""#@title 10. Exportar las sesiones de la app (decisiones, resúmenes y verificación)
requiere("cfg", "store")
import json, shutil
from datetime import datetime
from p2 import export

sesiones = sorted(p for p in store.root.iterdir() if p.is_dir()) if store.root.exists() else []
if not sesiones:
    print("⏸ Aún no hay sesiones: use la app de la celda 9 y vuelva a ejecutar esta celda.")
else:
    filas = []
    for d in sesiones:
        s = json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else None
        filas.append({"sesion": d.name, "con_resumen": s is not None,
                      "verificado": s and s["verified"], "fallback": s and s["fallback"], "intentos": s and s["attempts"]})
    print(json.dumps(filas, indent=1, ensure_ascii=False))
    con = [f for f in filas if f["con_resumen"]]
    if con:
        print(f"Resúmenes verificados: {sum(bool(f['verificado']) for f in con)}/{len(con)} | "
              f"con plantilla de respaldo: {sum(bool(f['fallback']) for f in con)}/{len(con)}")
    if "db" in globals():
        sus_rows = db.rows("sus")
        print(f"BD temporal: {len(db.rows('resumenes'))} respuestas del agente | {len(sus_rows)} cuestionarios SUS"
              + (f" | SUS medio {sum(r['puntuacion'] for r in sus_rows) / len(sus_rows):.1f}/100" if sus_rows else ""))
    export.write_manifest(store.root)
    zip_out = Path("/content") / f"sesiones_app_{datetime.now():%Y%m%d-%H%M}.zip"
    export.zip_dir(store.root, zip_out)
    print("ZIP:", zip_out)
    from google.colab import files
    files.download(str(zip_out))"""),

(CODE, r"""#@title 11. Borrar los datos temporales de la prueba (sesiones, imágenes y BD)
import shutil
from pathlib import Path

if "demo" in globals():
    demo.close()  # cierra la app y el enlace público antes de borrar
    print("App y enlace público cerrados.")
root = Path("/content") / cfg["app"]["sessions_dir"] if "cfg" in globals() else Path("/content/sesiones")
borrados = []
if root.exists():
    shutil.rmtree(root)
    borrados.append(str(root))
for z in Path("/content").glob("sesiones_app_*.zip"):
    z.unlink()
    borrados.append(str(z))
print("Borrado:", borrados or "nada (no había datos)")
print("Recuerde: el ZIP descargado en su PC NO se borra; trátelo como dato personal si contiene caras reales.")"""),

(MD, r"""## Después del piloto (en Windows)
1. Descomprima el ZIP de resultados en `results/`.
2. Evaluación ciega: `python -m experiments.blind_eval results/<carpeta> --evaluador <nombre>` (no abra `blind_key.json` antes de terminar).
3. Complete `LIMITATIONS.md` y compare A1 vs A2 con la regla de `planificacion/04` §8 (exploratorio)."""),
]


GUARD = ('if "requiere" not in globals():  # p.ej. tras reiniciar el entorno de ejecución\n'
         '    raise SystemExit("Ejecute las celdas en orden desde la 1 (Entorno de ejecución › Ejecutar todas).")\n')


def _with_guard(src):
    """Inserta GUARD tras la línea #@title en las celdas posteriores a la 1."""
    first, _, rest = src.partition("\n")
    return f"{first}\n{GUARD}{rest}"


# Celdas del piloto: sin autorización NO lanzan excepciones (Colab las muestra como error);
# se omiten con un aviso y «Ejecutar todas» termina limpio.
SKIP = {
    "#@title 6.": ("PILOTO_AUTORIZADO",
                   "⏸ Piloto NO ejecutado: falta autorización. Las comprobaciones (celdas 1–5) terminaron bien. "
                   "Para ejecutarlo, marque AUTORIZO_DESCARGAR_MODELOS y AUTORIZO_EJECUTAR_PILOTO en la celda 1 "
                   "y use «Entorno de ejecución › Ejecutar todas»."),
    "#@title 7.": ("PILOTO_AUTORIZADO", "⏸ Métricas omitidas: el piloto no se ejecutó (falta autorización en la celda 1)."),
    "#@title 8.": ("PILOTO_AUTORIZADO", "⏸ Exportación omitida: el piloto no se ejecutó (falta autorización en la celda 1)."),
    "#@title 9.": ("APP_AUTORIZADA",
                   "⏸ App no iniciada: falta autorización. Para usarla, marque AUTORIZO_DESCARGAR_MODELOS y AUTORIZO_APP "
                   "en la celda 1 y use «Entorno de ejecución › Ejecutar todas»."),
    "#@title 10.": ("APP_AUTORIZADA", "⏸ Exportación de sesiones omitida: la app no está autorizada (celda 1)."),
    "#@title 11.": ("CONFIRMO_BORRAR_DATOS",
                    "⏸ No se borró nada. Al terminar la prueba, descargue el ZIP (celda 10), marque CONFIRMO_BORRAR_DATOS "
                    "en la celda 1 y ejecute solo esta celda."),
}


def _pilot_gate(src):
    first, _, body = src.partition("\n")
    hit = next((v for t, v in SKIP.items() if first.startswith(t + " ")), None)
    if hit is None:
        return src
    var, msg = hit
    indented = "\n".join(("    " + l) if l.strip() else l for l in body.splitlines())
    return f"{first}\nif not {var}:\n    print({msg!r})\nelse:\n{indented}"


def p2_version():
    src = (ROOT / "p2" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', src, re.M).group(1)


def build(path="run_colab.ipynb"):
    sys.path.insert(0, str(ROOT))
    from tools.make_zip import data_problems, payload_bytes
    probs = data_problems()  # nada sin registrar (p.ej. fotos de terceros) puede acabar incrustado en el notebook
    if probs:
        raise SystemExit("Notebook no generado; data/sfhq_t2i no es coherente:\n- " + "\n- ".join(probs))
    payload = payload_bytes()
    subs = {"__P2_VERSION__": p2_version(), "__PAYLOAD_SHA256__": hashlib.sha256(payload).hexdigest(),
            "__PAYLOAD_B64__": base64.b64encode(payload).decode("ascii")}
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        for k, v in subs.items():
            src = src.replace(k, v)
        if kind == CODE and not src.lstrip().startswith(("#@title 1.", "#@title 1b.")):
            src = _with_guard(_pilot_gate(src.strip("\n")))
        meta = {"cellView": "form"} if kind == CODE else {}  # Colab muestra solo el formulario/título
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": meta, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build(ROOT / "run_colab.ipynb"))
