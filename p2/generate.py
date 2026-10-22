"""Generación de exactamente 5 variaciones. Backends: 'mock' (local, sin modelos) y 'sd15_img2img' (Colab)."""
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def plan(cfg, method):
    """Especificación de las 5 variaciones de un método (id, atributo, prompt, semilla)."""
    m = cfg["methods"][method]
    specs = [{"id": f"V{k}", "method": method, "attribute": v["attribute"], "prompt": v["prompt"],
              "seed": cfg["seed_base"] + k}
             for k, v in enumerate(m["variations"], start=1)]
    assert len(specs) == cfg["num_variations"] == 5
    return specs


# Línea base clásica / backend de prueba: transformaciones deterministas, sin modelos.
_MOCK = [
    lambda im: ImageEnhance.Color(im).enhance(1.6),
    lambda im: ImageOps.colorize(ImageOps.grayscale(im), black="#001830", white="#e0f0ff"),
    lambda im: im.filter(ImageFilter.SMOOTH_MORE).quantize(16).convert("RGB"),
    lambda im: ImageEnhance.Brightness(im).enhance(1.25),
    lambda im: ImageOps.grayscale(im).convert("RGB"),
]


def mock_generate(img, spec):
    k = int(spec["id"][1:])
    return _MOCK[k - 1](img.convert("RGB")), False


def load_sd15(cfg, device="cuda"):
    import torch

    from .compat import neutralize_incompatible_torchao
    print(neutralize_incompatible_torchao())
    from diffusers import AutoPipelineForImage2Image, EulerDiscreteScheduler
    if not cfg["generation"]["safety_checker"]:
        raise ValueError("El safety checker debe permanecer activo (decisión ética del proyecto).")
    m = cfg["models"]["sd15"]
    pipe = AutoPipelineForImage2Image.from_pretrained(
        m["repo"], revision=m["revision"], variant=m["variant"], torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)
    return pipe.to(device)


def sd15_generate(pipe, img, spec, gen):
    import torch
    out = pipe(prompt=spec["prompt"], negative_prompt=gen["negative_prompt"], image=img,
               strength=gen["strength"], num_inference_steps=gen["steps"], guidance_scale=gen["guidance_scale"],
               generator=torch.Generator("cuda").manual_seed(spec["seed"]))
    nsfw = bool(out.nsfw_content_detected[0]) if getattr(out, "nsfw_content_detected", None) else False
    return out.images[0], nsfw


def cinco_variaciones(img, specs, generate_one):
    """Aplica `generate_one(img, spec) -> (PIL, nsfw)` a las 5 specs. Nunca sustituye fallos en silencio."""
    if len(specs) != 5:
        raise ValueError(f"Se requieren exactamente 5 variaciones, hay {len(specs)}.")
    results = [(spec, *generate_one(img, spec)) for spec in specs]
    assert len(results) == 5
    return results
