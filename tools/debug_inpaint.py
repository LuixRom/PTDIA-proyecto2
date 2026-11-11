"""Depura un look: guarda el recorte, la máscara y el control que recibe el modelo, y su salida cruda.

Uso:  .venv\\Scripts\\python.exe tools\\debug_inpaint.py CARPETA_SESION SALIDA_DIR [--mask mask_superior_mangas]
      [--prompt "..."]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sesion")
    ap.add_argument("salida")
    ap.add_argument("--mask", default="mask_superior_mangas")
    ap.add_argument("--prompt", default="a person wearing an elegant long-sleeve red satin evening dress, realistic fashion photo")
    ap.add_argument("--guidance", type=float, default=None)
    a = ap.parse_args()
    import torch
    from PIL import Image

    from p2 import config, tryon
    cfg = config.load("config/vitrina.yaml")
    if a.guidance:
        cfg["generation"]["guidance_scale"] = a.guidance
    d, out = Path(a.sesion), Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(d / "input.png").convert("RGB")
    mask = Image.open(d / f"{a.mask}.png").convert("L")
    control = Image.open(d / "control.png").convert("RGB") if (d / "control.png").exists() else None
    pipe = tryon.load_inpaint(cfg, offload=True)
    spec = {"prompt": a.prompt, "seed": 1001}
    b = cfg["approaches"]["B"]

    def run(crop, mcrop, size, ccrop):
        crop.save(out / "1_recorte.png")
        mcrop.save(out / "2_mascara.png")
        if ccrop is not None:
            ccrop.save(out / "3_control.png")
        kw = {}
        if getattr(pipe, "controlnet", None) is not None:
            kw = dict(control_image=ccrop if ccrop is not None else Image.new("RGB", size),
                      controlnet_conditioning_scale=b["controlnet"]["conditioning_scale"],
                      control_guidance_end=b["controlnet"].get("guidance_end", 1.0))
        o = pipe(prompt=tryon.positive_for(spec, cfg), negative_prompt=tryon.negative_for(spec, cfg), image=crop,
                 mask_image=mcrop, strength=b["strength"], num_inference_steps=b["steps"],
                 guidance_scale=cfg["generation"]["guidance_scale"], width=size[0], height=size[1],
                 generator=torch.Generator("cuda").manual_seed(1001), **kw)
        o.images[0].save(out / "4_salida_cruda.png")
        print("strength", b["strength"], "| mask modo", mcrop.mode, "| valores", sorted(set(mcrop.getdata()))[:5])
        return o.images[0], False
    res, _ = tryon.inpaint_region(run, img, mask, b["work_size"], control)
    res.save(out / "5_final.png")


if __name__ == "__main__":
    main()
