"""Genera el catálogo sintético para entrenar el LoRA «Estilo de la tienda» (nivel avanzado).

Uso (laptop, modelos ya descargados; sin internet):
    .venv\\Scripts\\python.exe tools\\gen_catalogo_lora.py [--linea mujer|hombre] [--solo m_blusa_blanca,...] [--n 12]
    .venv\\Scripts\\python.exe tools\\gen_catalogo_lora.py --lista        (solo muestra los prompts, no genera)

Por cada prenda guarda imágenes + subtítulos (.txt con la palabra clave vtr_<id>), una hoja de contacto para
curar a mano (borrar las malas) y manifest.jsonl (prompt y semilla de cada imagen: reproducible).
Todo va FUERA del repositorio (salida en config/lora_catalogo.yaml). Las imágenes son sintéticas.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def prompts_for(cat, linea, pid, n):
    """[(prompt_de_imagen, subtítulo)] para n imágenes de una prenda. Sin modelos: se prueba en local."""
    g, item = cat[linea]["genero"], cat[linea]["prendas"][pid]
    tpl, var = cat["plantillas"][item["tipo"]], cat["variaciones"][item["tipo"]]
    out = []
    for k in range(n):
        v = var[k % len(var)]
        base = tpl.format(genero=g, prompt=item["prompt"])
        # subtítulo: la palabra clave + la descripción de la prenda (lo que el LoRA debe asociar)
        out.append((f"{base}, {v}", f"vtr_{pid}, {g} model wearing {item['prompt']}, {v}"))
    for k in range(item.get("producto_n", 0)):  # fotos de producto: solo el calzado, sin piernas ni poses
        v = cat["variaciones"]["producto"][k % len(cat["variaciones"]["producto"])]
        out.append((f"{cat['plantillas']['producto'].format(prompt=item['prompt'])}, {v}",
                    f"vtr_{pid}, product photo of {item['prompt']}, {v}"))
    return out


def seed_for(pid, k, base=7000):
    return base + int(hashlib.sha256(pid.encode()).hexdigest()[:6], 16) % 100000 + k


def sheet(paths, out, cols=6, h=260):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
    w = max(i.width for i in ims)
    rows = (len(ims) + cols - 1) // cols
    c = Image.new("RGB", (cols * (w + 4), rows * (h + 4)), "white")
    for n, i in enumerate(ims):
        c.paste(i, ((n % cols) * (w + 4), (n // cols) * (h + 4)))
    c.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linea", choices=["mujer", "hombre"])
    ap.add_argument("--solo", default="")
    ap.add_argument("--n", type=int)
    ap.add_argument("--lista", action="store_true")
    ap.add_argument("--piloto", action="store_true", help="solo las prendas de config piloto (24 imágenes c/u)")
    a = ap.parse_args()
    import yaml
    cat = yaml.safe_load((ROOT / "config/lora_catalogo.yaml").read_text(encoding="utf-8"))
    n = a.n or cat["imagenes_por_prenda"]
    only = {x.strip() for x in a.solo.split(",") if x.strip()}
    lineas = [a.linea] if a.linea else ["mujer", "hombre"]
    if a.piloto:
        pl = cat["piloto"]
        n, only, lineas = a.n or pl["imagenes_por_prenda"], set(pl["prendas"]), [pl["linea"]]
    jobs = [(l, p) for l in lineas for p in cat[l]["prendas"] if not only or p in only]
    if a.lista:
        for l, p in jobs:
            print(l, p, "->", prompts_for(cat, l, p, 1)[0][0])
        print(len(jobs), "prendas x", n, "=", len(jobs) * n, "imágenes")
        return

    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(cat["base"], torch_dtype=torch.float16, variant="fp16",
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()  # GPU de 4 GB
    pipe.set_progress_bar_config(disable=True)
    h, w = cat["size"]
    for l, p in jobs:
        d = Path(cat["salida"]) / l / p
        d.mkdir(parents=True, exist_ok=True)
        files = []
        with open(Path(cat["salida"]) / l / "manifest.jsonl", "a", encoding="utf-8") as mf:
            for k, (prompt, caption) in enumerate(prompts_for(cat, l, p, n)):
                f = d / f"{k:02d}.png"
                if f.exists():
                    files.append(f)
                    continue
                seed = seed_for(p, k)
                img = pipe(prompt, negative_prompt=cat["negative"], height=h, width=w, num_inference_steps=cat["steps"],
                           guidance_scale=cat["guidance"], generator=torch.Generator("cuda").manual_seed(seed)).images[0]
                img.save(f)
                f.with_suffix(".txt").write_text(caption, encoding="utf-8")
                mf.write(json.dumps({"file": f"{p}/{f.name}", "prompt": prompt, "caption": caption, "seed": seed},
                                    ensure_ascii=False) + "\n")
                files.append(f)
                print(f"{l}/{p} {k + 1}/{n}", flush=True)
        sheet(files, Path(cat["salida"]) / l / f"hoja_{p}.png")


if __name__ == "__main__":
    main()
