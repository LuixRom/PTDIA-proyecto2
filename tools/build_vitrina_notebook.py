"""Genera vitrina_colab.ipynb: el probador virtual Vitrina IA en Colab. Uso: python tools/build_vitrina_notebook.py

Reutiliza de tools/build_notebook.py el código incluido (celda 1b), su verificación (celda 2) y las guardas
de orden. Las celdas que descargan modelos se omiten con un aviso si falta la autorización.
"""
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import build_notebook as bn  # noqa: E402

MD, CODE = bn.MD, bn.CODE
_BY_TITLE = {src.lstrip().split("\n", 1)[0]: src for kind, src in bn.CELLS if kind == CODE}
CELL_1B = _BY_TITLE["#@title 1b. Código del proyecto incluido (no editar)"]
CELL_2 = _BY_TITLE["#@title 2. Preparar el código del proyecto y verificar su integridad"].replace(
    """try:
    import google.colab  # noqa: F401
except ImportError:
    raise SystemExit("Este notebook está pensado para Google Colab. En Windows use los tests locales (README.md).")""",
    """if not (EN_COLAB or EN_KAGGLE):
    raise SystemExit("Este notebook funciona en Google Colab o en Kaggle. En Windows use los tests locales (README.md).")""").replace(
    'zip_path = Path("/content/p2_codigo_incluido.zip")', 'zip_path = Path(BASE) / "p2_codigo_incluido.zip"')
CELL_3 = _BY_TITLE["#@title 3. Comprobar GPU, VRAM, RAM, disco y versiones (no instala nada)"].replace(
    "config/pilot.yaml", "config/vitrina.yaml").replace('env_check.collect("/content")', "env_check.collect(BASE)")
CELL_4 = _BY_TITLE["#@title 4. Validación local de compatibilidad (sin red, sin pip, no instala nada)"].replace(
    "las celdas 6 y 7 quedarán bloqueadas", "la celda 5 quedará bloqueada")

CELLS = [
(MD, r"""# Vitrina IA — Probador virtual de estilos (Proyecto 2, CS5363)

El cliente de una tienda de moda en línea sube **su foto** y se ve con **5 looks**. Elige la ocasión, prendas del catálogo o escribe su idea. Después:
- acepta o rechaza cada look, uno por uno o el paquete completo;
- pide una **nueva ronda** solo para los rechazados;
- puede **restaurar** cualquier configuración del historial;
- recibe un resumen de su elección, redactado por **uno de dos LLM** y comprobado por un verificador determinista.

**Cómo usarlo en Google Colab**
1. Active la GPU: *Entorno de ejecución › Cambiar tipo de entorno de ejecución › **GPU T4***.
2. En la celda 1 marque `AUTORIZO_DESCARGAR_MODELOS` y `AUTORIZO_APP`.
3. Ejecute *Entorno de ejecución › **Ejecutar todas***.

**Cómo usarlo en Kaggle** (respaldo si Colab agota la cuota de GPU)
1. En kaggle.com: *Create › New Notebook › File › Import Notebook* y suba este archivo.
2. En el panel derecho, *Session options*: **Accelerator → GPU T4 x2** e **Internet → On** (Kaggle pide verificar el teléfono una vez).
3. En la celda 1 marque `AUTORIZO_DESCARGAR_MODELOS` y `AUTORIZO_APP` y use *Run All*.
4. La app abre siempre con **enlace público** (el visor de Kaggle no muestra la app dentro de la celda). Los ZIP quedan en la pestaña *Output*.

Sin autorizaciones, las celdas 1–4 solo comprueban el entorno y el resto muestra «⏸ omitida»; no aparece ningún error.

**Generación — nivel intermedio** (el segundo enfoque se agregará después)
1. La segmentación de la ropa (SegFormer) crea una máscara según el look: solo el torso y los brazos, o también las piernas si el look incluye pantalón, terno o vestido.
2. Esa zona se recorta, se amplía a 768 px y se regenera con SD 1.5 *Inpainting* + **ControlNet openpose**: el esqueleto del cuerpo (ViTPose) fija la pose sin copiar texturas. Rostro y cabello nunca se repintan. Los brazos solo se repintan si el look es de **manga larga**, y las manos (ubicadas desde las muñecas) se protegen siempre.
3. Se pega de vuelta sobre la foto: el rostro, el cabello y el fondo se conservan.

**Guardrails:**
- Consentimiento y mayoría de edad, y un solo rostro por foto.
- Filtro NSFW sobre la entrada y las salidas.
- Lista de términos vetados (desvestir, cambiar el cuerpo, menores, violencia y marcas), aplicada también a la traducción de la instrucción.
- Comprobación de identidad: SSIM del rostro, con un reintento.
- Etiqueta «Simulación IA» en cada imagen; la foto no se guarda.

**Semillas:** en *modo prueba* son fijas (reproducibles); en *producción* cambian en cada ronda y quedan en `history.jsonl`.

**Descargas (con autorización, estimadas):**

| Modelo | Tamaño aprox. |
|---|---|
| SD 1.5 Inpainting | 2–4 GB |
| ControlNet openpose | 0,7–1,4 GB |
| SegFormer | 0,1 GB |
| ViTPose (pose del cuerpo) | 0,4 GB |
| CLIP | 0,6 GB |
| Qwen2.5-1.5B | 3,1 GB |
| Phi-3.5-mini (solo si se elige en la app) | 7,6 GB |

**No instala nada:** usa los paquetes que trae Colab o Kaggle. Si Colab trae versiones más nuevas que las probadas, la celda 4 lo muestra como aviso (⚠) y **continúa**. Solo se detiene si falta un paquete, si la versión es menor que la mínima o si la importación real falla."""),

(CODE, r"""#@title 1. Parámetros y autorizaciones
AUTORIZO_DESCARGAR_MODELOS = False  #@param {type:"boolean"}
#@markdown App del probador (celda 5). Requiere también AUTORIZO_DESCARGAR_MODELOS.
AUTORIZO_APP = False  #@param {type:"boolean"}
#@markdown Enlace público temporal *.gradio.live para la demo y las pruebas con usuarios (máx. 72 h).
ENLACE_PUBLICO = False  #@param {type:"boolean"}
#@markdown Enfoque intermedio (B). El básico (A, img2img) queda desactivado hasta que se retome.
CARGAR_ENFOQUE_A = False  #@param {type:"boolean"}
CARGAR_ENFOQUE_B = True  #@param {type:"boolean"}
CARGAR_METRICAS_CLIP = True  #@param {type:"boolean"}
#@markdown Mini-benchmark (celda 6): fotos de prueba de personas ADULTAS que dieron su consentimiento por escrito.
AUTORIZO_BENCHMARK = False  #@param {type:"boolean"}
CONFIRMO_CONSENTIMIENTO_FOTOS = False  #@param {type:"boolean"}
BENCH_LINEA = "mujer"  #@param ["mujer", "hombre"]
BENCH_OCASION = "reunion"  #@param ["trabajo", "reunion", "evento", "casual"]
#@markdown Al terminar (tras descargar el ZIP de la celda 7): borra sesiones, imágenes y la BD temporal.
CONFIRMO_BORRAR_DATOS = False  #@param {type:"boolean"}
FUENTE_CODIGO = "incluido"  #@param ["incluido", "subir", "drive"]
RUTA_ZIP_EN_DRIVE = "MyDrive/proyecto2/proyecto2_colab.zip"  #@param {type:"string"}
#@markdown Mini-benchmark: carpeta con las fotos de prueba (en Kaggle puede ser un dataset: /kaggle/input/<nombre>).
CARPETA_FOTOS_BENCH = ""  #@param {type:"string"}
import os
try:
    import google.colab  # noqa: F401
    EN_COLAB = True
except ImportError:
    EN_COLAB = False
EN_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))
BASE = "/kaggle/working" if EN_KAGGLE else "/content"   # carpeta de trabajo de cada plataforma
WORKDIR = f"{BASE}/p2run"
VALIDATION_PATH = "env/validation_vitrina.json"
EXPECTED_P2_VERSION = "__P2_VERSION__"  # fijado al generar el notebook


def requiere(*nombres):
    # Detiene la celda con una explicación si no se ejecutaron antes las celdas previas.
    faltan = [n for n in nombres if n not in globals()]
    if faltan:
        raise SystemExit(f"Falta ejecutar celdas anteriores (no existe: {', '.join(faltan)}). "
                         "Use «Entorno de ejecución › Ejecutar todas» o ejecute las celdas desde la 1 en orden.")


APP_AUTORIZADA = bool(AUTORIZO_DESCARGAR_MODELOS and AUTORIZO_APP)
BENCH_AUTORIZADO = bool(APP_AUTORIZADA and AUTORIZO_BENCHMARK and CONFIRMO_CONSENTIMIENTO_FOTOS)
print("Plataforma:", "Kaggle" if EN_KAGGLE else "Colab" if EN_COLAB else "desconocida", "| carpeta:", BASE)
print("Autorizaciones:", dict(descargar=AUTORIZO_DESCARGAR_MODELOS, app=AUTORIZO_APP, publico=ENLACE_PUBLICO,
                              benchmark=BENCH_AUTORIZADO, codigo=FUENTE_CODIGO))"""),

(CODE, CELL_1B),
(CODE, CELL_2),
(CODE, CELL_3),
(CODE, CELL_4),

(CODE, r"""#@title 5. Cargar modelos y abrir el probador virtual
requiere("cfg")
try:
    validation = env_check.require_validation(cfg, VALIDATION_PATH)
except env_check.ValidationError as e:
    raise SystemExit(f"Bloqueado: {e}")
try:
    import gradio  # noqa: F401  (preinstalado en Colab)
except ImportError:
    raise SystemExit("gradio no está instalado en este entorno; este notebook no instala paquetes.")
if not (CARGAR_ENFOQUE_A or CARGAR_ENFOQUE_B):
    raise SystemExit("Marque al menos un enfoque (CARGAR_ENFOQUE_A o CARGAR_ENFOQUE_B) en la celda 1.")
import gc, torch
import p2
from p2 import generate, metrics, tryon, vitrina, vitrina_ui, llm as p2llm
from p2.decisions import DecisionStore
from p2.guardrails import GuardrailError
from p2.tempdb import TempDB

if "demo" in globals():
    demo.close()  # al repetir la celda: cierra la app anterior antes de cargar otra vez
gc.collect(); torch.cuda.empty_cache()
pipe_a = generate.load_sd15(cfg) if CARGAR_ENFOQUE_A else None
pipe_b = tryon.load_inpaint(cfg) if CARGAR_ENFOQUE_B else None
segmenter = tryon.Segmenter(cfg)
pose = tryon.PoseEstimator(cfg)  # esqueleto: ControlNet openpose y ubicación de las manos
clip = metrics.Models(cfg, use_lpips=False) if CARGAR_METRICAS_CLIP else None


def gen_a(img, spec):
    if pipe_a is None:
        raise GuardrailError("El enfoque A no está cargado en este Colab (celda 1: CARGAR_ENFOQUE_A).")
    return tryon.gen_a(pipe_a, img, spec, cfg)


def gen_b(img, mask, spec, control=None):
    if pipe_b is None:
        raise GuardrailError("El enfoque B no está cargado en este Colab (celda 1: CARGAR_ENFOQUE_B).")
    return tryon.gen_b(pipe_b, img, mask, spec, cfg, control)


model_ids = {"A": cfg["models"]["sd15"]["revision"] if pipe_a else None,
             "B": tryon.model_commit(pipe_b) if pipe_b else None, "segmenter": segmenter.commit,
             "controlnet": tryon.model_commit(pipe_b.controlnet) if getattr(pipe_b, "controlnet", None) else None}
engine = vitrina.Engine(cfg, segmenter, gen_a, gen_b, lambda img: tryon.nsfw_input(pipe_b or pipe_a, img), clip,
                        model_ids={**model_ids, "pose": pose.commit}, pose=pose.detect)
llms = vitrina_ui.LLMs(cfg, p2llm.load_llm_entry)
print("Cargando el LLM por defecto (Qwen2.5-1.5B)…")
llms.get("qwen")
store = DecisionStore(Path(BASE) / cfg["app"]["sessions_dir"])
db = TempDB(store.root / "agente_tmp.sqlite")
demo = vitrina_ui.build_ui(store, cfg, engine, llms, db)
print("Probador versión", p2.__version__, "(debe coincidir con la cabecera de la app) | Modelos:", model_ids, "| VRAM en uso:", f"{torch.cuda.memory_allocated() / 1e9:.1f} GB")
PUBLICO = bool(ENLACE_PUBLICO or EN_KAGGLE)  # Kaggle no muestra la app dentro de la celda: siempre con enlace
demo.launch(share=PUBLICO, inline=not PUBLICO, height=1500, quiet=True, show_error=True,
            allowed_paths=[str(store.root)], **vitrina_ui.launch_kwargs())
if PUBLICO:
    print("ENLACE PÚBLICO TEMPORAL:", demo.share_url)
    print("Funciona mientras esta sesión siga abierta. Al terminar: celda 7 (descargar) y celda 8 (borrar).")"""),

(CODE, r"""#@title 6. Mini-benchmark del enfoque intermedio y comparación de los 2 LLM (fotos con consentimiento)
requiere("engine", "llms", "store")
import csv, json
from p2.ui import on_submit

BENCH = Path(BASE) / "benchmark_vitrina"
CARPETA = Path(CARPETA_FOTOS_BENCH) if CARPETA_FOTOS_BENCH else BENCH / "fotos"
CARPETA.mkdir(parents=True, exist_ok=True) if not CARPETA_FOTOS_BENCH else None
fotos = sorted(p for p in CARPETA.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")) \
    if CARPETA.exists() else []
if not fotos:  # nunca se queda esperando una subida ni lanza error: avisa y la celda no hace nada más
    print(f"⏸ No hay fotos en {CARPETA}. Súbalas ahí (panel Archivos en Colab; en Kaggle «Add Input» y ponga su ruta "
          "en CARPETA_FOTOS_BENCH) y vuelva a ejecutar solo esta celda. Solo personas adultas con consentimiento escrito.")
else:
    print(f"{len(fotos)} fotos en {CARPETA}")
bstore = DecisionStore(BENCH / "sesiones")
filas, sesiones = [], []
for foto in fotos:
    for enfoque in [k for k, ok in (("A", CARGAR_ENFOQUE_A), ("B", CARGAR_ENFOQUE_B)) if ok]:
        try:
            sid, _ = vitrina.start_session(bstore, engine, foto.read_bytes(), dict(
                mode="ocasion", occasion=BENCH_OCASION, linea=BENCH_LINEA, approach=enfoque, seed_mode="prueba",
                own_photo=True, adult=True, llm="qwen"))
        except GuardrailError as e:
            print(f"{foto.name} / {enfoque}: bloqueada por guardrail → {e}")
            filas.append({"foto": foto.name, "enfoque": enfoque, "bloqueada": str(e)})
            continue
        s = bstore.load(sid)
        h = vitrina.history(bstore, sid)[-1]
        for v in s.variations:
            filas.append({"foto": foto.name, "enfoque": enfoque, "id": v.id, "look": v.attribute, "seed": v.seed,
                          **{k: v.metrics.get(k) for k in ("face_ssim", "identity_ok", "clip_delta", "nsfw", "attempts", "time_s")},
                          **{f"set_{k}": x for k, x in h["set"].items()}})
        sesiones.append(sid)
with open(BENCH / "benchmark_imagenes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in filas for k in r}))
    w.writeheader(); w.writerows(filas)
for enfoque in [k for k, ok in (("A", CARGAR_ENFOQUE_A), ("B", CARGAR_ENFOQUE_B)) if ok]:
    xs = [r for r in filas if r.get("enfoque") == enfoque and "id" in r]
    if xs:
        def mean(k, fmt):  # "n/d" si la métrica no existe (p. ej. no se detectó rostro)
            vals = [r[k] for r in xs if r.get(k) is not None]
            return format(sum(vals) / len(vals), fmt) if vals else "n/d"
        print(f"Enfoque {enfoque}: n={len(xs)} | SSIM rostro {mean('face_ssim', '.3f')} | ΔCLIP {mean('clip_delta', '+.4f')} | "
              f"{mean('time_s', '.1f')} s/img | NSFW {sum(bool(r['nsfw']) for r in xs)} | "
              f"identidad OK {sum(r['identity_ok'] is True for r in xs)}/{sum(r['identity_ok'] is not None for r in xs)} con rostro")

# Comparación de LLM: los mismos patrones de decisión (incluidos todo aceptado y todo rechazado) para ambos.
PATRONES = [["aceptada"] * 5, ["rechazada"] * 5, ["aceptada", "rechazada", "aceptada", "rechazada", "aceptada"],
            ["rechazada", "aceptada", "rechazada", "aceptada", "rechazada"], ["aceptada", "aceptada", "rechazada", "rechazada", "aceptada"]]
MOTIVOS = ["me queda formal", "no es mi estilo", "", "el color no me favorece", ""]
llm_filas = []
for key in (cfg["llms"] if sesiones else []):  # sin fotos no se carga ningún LLM extra
    fn, label = llms.get(key)
    for i, sid in enumerate(sesiones[:len(PATRONES)]):
        on_submit(bstore, sid, PATRONES[i], MOTIVOS)
        res = vitrina.summarize(bstore, sid, fn, label)
        llm_filas.append({"llm": label, "sesion": sid, "patron": i, "verificado_llm": res.verified and not res.fallback,
                          "intentos": res.attempts, "respaldo": res.fallback, "resumen": res.summary.resumen})
llms.get("qwen")  # deja cargado el LLM por defecto de la app
with open(BENCH / "benchmark_llm.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(llm_filas[0]) if llm_filas else ["llm"])
    w.writeheader(); w.writerows(llm_filas)
for key in cfg["llms"]:
    xs = [r for r in llm_filas if r["llm"].startswith(cfg["llms"][key]["repo"])]
    if xs:
        print(f"{cfg['llms'][key]['name']}: resumen propio verificado {sum(r['verificado_llm'] for r in xs)}/{len(xs)} | "
              f"intentos medios {sum(r['intentos'] for r in xs) / len(xs):.1f} | plantilla {sum(r['respaldo'] for r in xs)}")
print("Exploratorio: pocas fotos y patrones; sin inferencia estadística. CSV en", BENCH)"""),

(CODE, r"""#@title 7. Exportar sesiones, historial, resúmenes y benchmark (ZIP)
requiere("cfg", "store")
import json, shutil
from datetime import datetime
from p2 import export

paquete = Path(BASE) / "export_vitrina"
if paquete.exists():
    shutil.rmtree(paquete)
paquete.mkdir()
for src in (store.root, Path(BASE) / "benchmark_vitrina"):
    if src.exists():  # sin fotos originales ni sus máscaras
        shutil.copytree(src, paquete / src.name,
                        ignore=shutil.ignore_patterns("input.png", "mask*.png", "control.png", "fotos"))
if not any(paquete.iterdir()):
    print("⏸ Aún no hay sesiones ni benchmark: use la app (celda 5) y vuelva a ejecutar esta celda.")
else:
    n = export.write_manifest(paquete)
    zip_out = Path(BASE) / f"vitrina_{datetime.now():%Y%m%d-%H%M}.zip"
    export.zip_dir(paquete, zip_out)
    print(f"{n} archivos (sin fotos de entrada) → {zip_out}")
    if EN_COLAB:
        from google.colab import files
        files.download(str(zip_out))
    else:
        print("En Kaggle: descárguelo desde la pestaña Output (panel derecho) o con *Save Version*.")"""),

(CODE, r"""#@title 8. Borrar los datos temporales (sesiones, fotos de benchmark, BD)
import shutil
from pathlib import Path

if "demo" in globals():
    demo.close()
    print("App y enlace público cerrados.")
borrados = []
BASE_DIR = Path(globals().get("BASE", "/content"))
for p in [BASE_DIR / "sesiones_vitrina", BASE_DIR / "benchmark_vitrina", BASE_DIR / "export_vitrina",
          *BASE_DIR.glob("vitrina_*.zip")]:
    if p.is_dir():
        shutil.rmtree(p); borrados.append(str(p))
    elif p.exists():
        p.unlink(); borrados.append(str(p))
print("Borrado:", borrados or "nada (no había datos)")
print("El ZIP descargado en su PC no se borra: contiene imágenes de personas, trátelo como dato personal.")"""),
]

SKIP = {
    "#@title 5.": ("APP_AUTORIZADA", "⏸ Probador no iniciado: marque AUTORIZO_DESCARGAR_MODELOS y AUTORIZO_APP en la "
                   "celda 1 y use «Entorno de ejecución › Ejecutar todas»."),
    "#@title 6.": ("BENCH_AUTORIZADO", "⏸ Mini-benchmark omitido: requiere la app (celda 5), AUTORIZO_BENCHMARK y "
                   "CONFIRMO_CONSENTIMIENTO_FOTOS en la celda 1."),
    "#@title 7.": ("APP_AUTORIZADA", "⏸ Exportación omitida: la app no está autorizada (celda 1)."),
    "#@title 8.": ("CONFIRMO_BORRAR_DATOS", "⏸ No se borró nada. Al terminar, descargue el ZIP (celda 7), marque "
                   "CONFIRMO_BORRAR_DATOS en la celda 1 y ejecute solo esta celda."),
}


def _gate(src):
    first, _, body = src.partition("\n")
    hit = next((v for t, v in SKIP.items() if first.startswith(t + " ")), None)
    if hit is None:
        return src
    var, msg = hit
    indented = "\n".join(("    " + l) if l.strip() else l for l in body.splitlines())
    return f"{first}\nif not {var}:\n    print({msg!r})\nelse:\n{indented}"


def build(path=ROOT / "vitrina_colab.ipynb"):
    from tools.make_zip import payload_bytes
    payload = payload_bytes()
    subs = {"__P2_VERSION__": bn.p2_version(), "__PAYLOAD_SHA256__": hashlib.sha256(payload).hexdigest(),
            "__PAYLOAD_B64__": base64.b64encode(payload).decode("ascii")}
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        for k, v in subs.items():
            src = src.replace(k, v)
        if kind == CODE and not src.lstrip().startswith(("#@title 1.", "#@title 1b.")):
            src = bn._with_guard(_gate(src.strip("\n")))
        meta = {"cellView": "form"} if kind == CODE else {}
        cell = {"cell_type": kind, "id": f"v{i:02d}", "metadata": meta, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return Path(path)


if __name__ == "__main__":
    print(build())
