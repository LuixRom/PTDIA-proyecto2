"""Genera lora_entrenamiento_kaggle.ipynb: entrena el LoRA «Estilo de la tienda» (nivel avanzado) en Kaggle o Colab.

Uso:  python tools/build_lora_notebook.py        ->  lora_entrenamiento_kaggle.ipynb (en la raíz del repositorio)

El notebook usa el script oficial de diffusers (train_text_to_image_lora.py, de la MISMA versión instalada) sobre
el paquete que arma tools/curar_lora.py (zip con imágenes + metadata.jsonl). Sin tokens ni claves: solo modelos abiertos.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIFFUSERS = "0.41.0"  # versión probada del proyecto (config/vitrina.yaml: dependencies.tested)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Vitrina IA · Nivel avanzado: LoRA «Estilo de la tienda» (piloto, línea mujer)

Entrena un **LoRA** sobre Stable Diffusion 1.5 con el catálogo sintético curado (76 imágenes: camisa celeste, blusa blanca,
tacones negros y zapatillas blancas). Cada prenda tiene una palabra clave (`vtr_m_...`) que el LoRA aprende a asociar con su aspecto.

**Antes de ejecutar (Kaggle):**
1. Sube `piloto_mujer.zip` como *Dataset* (Create → New Dataset). Kaggle lo descomprime solo.
2. En este notebook: *Add Input* → tu dataset. *Settings* → **Accelerator: GPU (T4 o P100)** y **Internet: On**.
3. Ejecuta las celdas en orden. Tarda de 45 a 90 minutos (estimado, no medido). Al final descarga `lora_piloto_mujer.zip` desde *Output*.

En Colab: sube `piloto_mujer.zip` al panel de archivos y ejecuta igual (GPU T4).

No se usa ninguna clave ni token. Las imágenes de entrenamiento son sintéticas (sin personas reales).
"""),
    code("""
#@title 1 · Parámetros
STEPS = 800              #@param {type:"integer"}   # pasos de optimización; con ~76 imágenes, menos es más (sobreajuste)
CHECKPOINT_EVERY = 200   #@param {type:"integer"}   # guarda un LoRA cada N pasos para elegir el mejor punto
RANK = 16                #@param {type:"integer"}   # capacidad del LoRA; bajo = memoriza menos
LEARNING_RATE = 1e-4     #@param {type:"number"}
SEED = 1337              #@param {type:"integer"}
RETRAIN = False          #@param {type:"boolean"}   # False: si ya hay un LoRA entrenado se omite el paso 5 (no lo borra)
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DIFFUSERS_VERSION = "%(diffusers)s"
"""),
    code("""
#@title 2 · Entorno
import os, subprocess, sys, glob, json, shutil, zipfile
from pathlib import Path

EN_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
WORK = Path("/kaggle/working" if EN_KAGGLE else "/content")


def run(cmd, **kw):
    \"\"\"Ejecuta y muestra la salida en vivo; falla si el comando falla.\"\"\"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    for line in p.stdout:
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"Falló: {' '.join(map(str, cmd))}")


# torchao (preinstalado en Colab) choca con diffusers: «cannot import name FqnToConfig». No se usa: se quita.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
run([sys.executable, "-m", "pip", "install", "-q", f"diffusers=={DIFFUSERS_VERSION}", "peft", "datasets", "accelerate", "transformers"])
import torch
assert torch.cuda.is_available(), "Activa una GPU (Kaggle: Settings → Accelerator; Colab: Entorno de ejecución → GPU)."
print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
import diffusers
print("diffusers", diffusers.__version__)
# Prueba temprana: si hay un choque de versiones falla aquí y no 30 minutos después, en el entrenamiento.
run([sys.executable, "-c", "from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel; import peft; print('imports OK')"])
"""),
    code("""
#@title 3 · Datos (zip subido como Dataset o al panel de archivos)
DATA = WORK / "datos"
shutil.rmtree(DATA, ignore_errors=True)
DATA.mkdir(parents=True)
metas = glob.glob("/kaggle/input/**/metadata.jsonl", recursive=True) + glob.glob("/content/**/metadata.jsonl", recursive=True)
zips = glob.glob("/kaggle/input/**/piloto_mujer.zip", recursive=True) + glob.glob("/content/piloto_mujer.zip")
if metas:  # Kaggle ya descomprimió el zip: se copia a una carpeta escribible
    src = Path(metas[0]).parent
    for f in src.iterdir():
        if f.suffix in {".png", ".jsonl"}:
            shutil.copy(f, DATA / f.name)
elif zips:
    zipfile.ZipFile(zips[0]).extractall(DATA)
else:
    raise SystemExit("No encuentro el dataset: sube piloto_mujer.zip (ver instrucciones arriba).")

rows = [json.loads(l) for l in (DATA / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
missing = [r["file_name"] for r in rows if not (DATA / r["file_name"]).exists()]
assert not missing, f"Faltan imágenes: {missing[:3]}"
assert len(rows) >= 20, f"Muy pocas imágenes ({len(rows)})"
print(len(rows), "imágenes con descripción. Ejemplo:", rows[0]["text"])
from collections import Counter
print(Counter(r["text"].split(",")[0] for r in rows))
"""),
    code("""
#@title 4 · Script oficial de entrenamiento (misma versión que diffusers)
SCRIPT = WORK / "train_text_to_image_lora.py"
url = f"https://raw.githubusercontent.com/huggingface/diffusers/v{DIFFUSERS_VERSION}/examples/text_to_image/train_text_to_image_lora.py"
run(["wget", "-q", "-O", str(SCRIPT), url])
src = SCRIPT.read_text(encoding="utf-8")
for arg in ("--rank", "--train_data_dir", "--checkpointing_steps", "--gradient_checkpointing", "--max_train_steps"):
    assert arg in src, f"El script descargado no tiene {arg}: revisa la versión de diffusers ({DIFFUSERS_VERSION})."
print("Script listo:", SCRIPT)
"""),
    code("""
#@title 5 · Entrenar (el paso largo)
OUT = WORK / "lora_piloto_mujer"
if (OUT / "pytorch_lora_weights.safetensors").exists() and not RETRAIN:
    print("Ya hay un LoRA entrenado en", OUT, "→ se omite el entrenamiento (RETRAIN=True para repetirlo).")
else:
  shutil.rmtree(OUT, ignore_errors=True)
  run(["accelerate", "launch", "--num_processes", "1", "--mixed_precision", "fp16", str(SCRIPT),
       "--pretrained_model_name_or_path", BASE_MODEL,
       "--train_data_dir", str(DATA), "--caption_column", "text",
       "--resolution", "512", "--random_flip",  # recorte aleatorio (por defecto): el calzado no se pierde siempre
       "--train_batch_size", "2", "--gradient_accumulation_steps", "2",
       "--max_train_steps", str(STEPS), "--checkpointing_steps", str(CHECKPOINT_EVERY),
       "--learning_rate", str(LEARNING_RATE), "--lr_scheduler", "cosine", "--lr_warmup_steps", "50",
       "--rank", str(RANK), "--mixed_precision", "fp16", "--gradient_checkpointing",
       "--seed", str(SEED), "--dataloader_num_workers", "2",
       "--output_dir", str(OUT)])
"""),
    code("""
#@title 6 · Verificar lo entrenado
ckpts = sorted(OUT.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
final = OUT / "pytorch_lora_weights.safetensors"
print("final:", final.exists(), f"{final.stat().st_size / 1e6:.1f} MB" if final.exists() else "")
for c in ckpts:
    w = c / "pytorch_lora_weights.safetensors"
    print(c.name, "LoRA:", w.exists(), "| contenido:", sorted(p.name for p in c.iterdir())[:4])
assert final.exists(), "No se generó el LoRA final: revisa el registro del paso 5."
"""),
    code("""
#@title 7 · Comparar: sin LoRA y con cada punto de control (mismas semillas)
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image, ImageDraw

pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, safety_checker=None,
                                               requires_safety_checker=False).to("cuda")
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)
PROMPTS = [
    "vtr_m_tacones_negros, woman model wearing plain all-black leather high heel pumps, front view",
    "vtr_m_zapatillas_blancas, woman model wearing plain all-white leather sneakers, front view",
    "vtr_m_camisa_celeste, woman model wearing a pale sky-blue button-up shirt with black trousers, front view",
    "vtr_m_blusa_blanca, woman model wearing a crisp white button-up blouse with black trousers, front view",
]
NEG = "text, watermark, deformed, extra limbs, blurry, lowres"
variants = [("sin LoRA", None)] + [(c.name, c) for c in ckpts] + [("final", OUT)]
S, rows_img = 256, []
for name, path in variants:
    pipe.unload_lora_weights()
    if path is not None:
        pipe.load_lora_weights(str(path))
    row = [pipe(p, negative_prompt=NEG, num_inference_steps=25, guidance_scale=7.5, height=384, width=384,
                generator=torch.Generator("cuda").manual_seed(11 + k)).images[0].resize((S, S)) for k, p in enumerate(PROMPTS)]
    rows_img.append((name, row))
sheet = Image.new("RGB", (len(PROMPTS) * S + 110, len(rows_img) * S), "white")
d = ImageDraw.Draw(sheet)
for r, (name, row) in enumerate(rows_img):
    d.text((4, r * S + S // 2), name, fill=(0, 0, 0))
    for c, im in enumerate(row):
        sheet.paste(im, (110 + c * S, r * S))
sheet.save(WORK / "comparacion_lora.png")
sheet
"""),
    code("""
#@title 8 · Empaquetar para descargar (Output → lora_piloto_mujer.zip)
ZIP = WORK / "lora_piloto_mujer.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(OUT / "pytorch_lora_weights.safetensors", "final/pytorch_lora_weights.safetensors")
    for c in ckpts:
        z.write(c / "pytorch_lora_weights.safetensors", f"{c.name}/pytorch_lora_weights.safetensors")
    z.write(WORK / "comparacion_lora.png", "comparacion_lora.png")
    z.writestr("parametros.json", json.dumps({"steps": STEPS, "checkpoint_every": CHECKPOINT_EVERY, "rank": RANK,
        "learning_rate": LEARNING_RATE, "seed": SEED, "base": BASE_MODEL, "diffusers": DIFFUSERS_VERSION,
        "imagenes": len(rows)}, indent=1))
print(ZIP, f"{ZIP.stat().st_size / 1e6:.1f} MB")
"""),
]


def build(path):
    nb = {"cells": [{**c, "source": [s % {"diffusers": DIFFUSERS} if "%(diffusers)s" in s else s for s in c["source"]]}
                    for c in CELLS],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return Path(path)


if __name__ == "__main__":
    print(build(ROOT / "lora_entrenamiento_kaggle.ipynb"))
