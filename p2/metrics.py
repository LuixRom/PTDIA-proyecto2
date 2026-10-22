"""Métricas de fidelidad, diversidad y calidad.

SSIM (scikit-image) y los chequeos de calidad son baratos y locales. LPIPS y CLIP
necesitan torch y descargar pesos: solo se cargan si se llama a `load_models` (Colab).
Ninguna métrica mide identidad: la identidad se juzga en la evaluación humana ciega.
"""
from itertools import combinations

import numpy as np


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(np.asarray(a), np.asarray(b), channel_axis=2, data_range=255))


def is_black(img, threshold=8):
    return float(np.asarray(img.convert("L")).mean()) < threshold


def clip_features(out):
    """transformers 4.x: get_*_features devuelve el tensor proyectado; 5.x: un BaseModelOutputWithPooling
    cuyo pooler_output es esa proyección. Acepta ambos."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def pairwise_mean(items, dist):
    vals = [dist(a, b) for a, b in combinations(items, 2)]
    return float(np.mean(vals)), len(vals)


class Models:
    """CLIP (y LPIPS si está instalado y se pide) en GPU. Se crea solo en Colab."""

    def __init__(self, cfg, device="cuda", use_lpips=True):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        c = cfg["models"]["clip"]
        self.torch, self.device = torch, device
        self.lpips = None
        if use_lpips:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.clip = CLIPModel.from_pretrained(c["repo"], revision=c["revision"]).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(c["repo"], revision=c["revision"])

    def lpips_dist(self, a, b):
        t = self.torch

        def tens(im):
            x = t.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)
            return x.unsqueeze(0).to(self.device)
        with t.no_grad():
            return float(self.lpips(tens(a), tens(b)).item())

    def img_emb(self, im):
        with self.torch.no_grad():
            e = clip_features(self.clip.get_image_features(**self.proc(images=im, return_tensors="pt").to(self.device)))
        return (e / e.norm(dim=-1, keepdim=True))[0]

    def txt_emb(self, text):
        with self.torch.no_grad():
            e = clip_features(self.clip.get_text_features(**self.proc(text=[text], return_tensors="pt", padding=True).to(self.device)))
        return (e / e.norm(dim=-1, keepdim=True))[0]


def per_image(orig, var, attribute_text=None, models=None, face_counter=None):
    m = {"ssim": ssim(orig, var), "black": is_black(var)}
    if face_counter is not None:
        m["faces"] = face_counter(var)
    if models is not None:
        eo, ev = models.img_emb(orig), models.img_emb(var)
        m["lpips"] = models.lpips_dist(orig, var) if models.lpips is not None else None  # None = no disponible
        m["clip_img_sim"] = float(eo @ ev)  # similitud semántica, NO identidad
        if attribute_text:
            t = models.txt_emb(attribute_text)
            m["clip_delta"] = float(ev @ t - eo @ t)
    return m


def per_set(images, models=None):
    s = {"ssim_div": pairwise_mean(images, lambda a, b: 1 - ssim(a, b))[0], "pairs": 10}
    if models is not None:
        embs = [models.img_emb(im) for im in images]
        s["lpips_div"] = pairwise_mean(images, models.lpips_dist)[0] if models.lpips is not None else None
        s["clip_div"] = pairwise_mean(embs, lambda a, b: 1 - float(a @ b))[0]
    return s
