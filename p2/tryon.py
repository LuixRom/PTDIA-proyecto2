"""Generación del probador virtual. Enfoque A: img2img de toda la foto (SD 1.5, el de la versión 0).
Enfoque B: segmentación de la ropa (SegFormer) + inpainting solo de esa región (SD 1.5 Inpainting).

Las funciones sobre mapas de etiquetas son puras (numpy/PIL) y se prueban en local; los modelos solo
se cargan en Colab. Las versiones `mock_*` reproducen la interfaz sin modelos (tests y modo --mock).
"""
import hashlib

import numpy as np
from PIL import Image, ImageFilter

from .metrics import ssim

# Etiquetas ATR del segmentador (id2label de mattmdjaga/segformer_b2_clothes); el mock usa las mismas.
MOCK_ID2LABEL = {0: "Background", 2: "Hair", 4: "Upper-clothes", 11: "Face", 14: "Left-arm", 15: "Right-arm"}


def _ids(id2label, names):
    wanted = {n.lower() for n in names}
    return [int(i) for i, n in id2label.items() if n.lower() in wanted]


def _grow(m, px):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(2 * px + 1))) > 127


def _shrink(m, px):
    return np.asarray(Image.fromarray((m * 255).astype(np.uint8), "L").filter(ImageFilter.MinFilter(2 * px + 1))) > 127


def mask_from_labels(lab, id2label, mask_names, protect_names, dilate_px=9, protect_extra=None, core_only=None):
    """Máscara de la ropa (blanco = se regenera). Se dilata para cubrir bordes y nunca toca lo protegido
    (rostro, cabello y, si se pasa `protect_extra`, p. ej. las manos). Devuelve (máscara L, fracción cubierta)."""
    m = np.isin(lab, _ids(id2label, mask_names))
    im = Image.fromarray((m * 255).astype(np.uint8), "L")
    if dilate_px > 0:
        im = im.filter(ImageFilter.MaxFilter(2 * (dilate_px // 2) + 1))
    m = np.asarray(im) > 127
    core_only = core_only or {}  # {etiqueta: px}: de esa etiqueta solo se protege el núcleo (p. ej. el cabello)
    m &= ~np.isin(lab, _ids(id2label, [n for n in protect_names if n not in core_only]))
    for name, px in core_only.items():
        if name in protect_names:
            m &= ~_shrink(np.isin(lab, _ids(id2label, [name])), px)
    if protect_extra is not None:
        m &= ~protect_extra
    return Image.fromarray((m * 255).astype(np.uint8), "L"), float(m.mean())


# Pose (COCO-17 de ViTPose): 5/6 hombros, 7/8 codos, 9/10 muñecas (izquierda/derecha de la persona).
ARMS = ((7, 9), (8, 10))  # (codo, muñeca)


def _disc(shape, cx, cy, rx, ry):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1


def accessory_masks(lab, id2label, kps, ac, min_score=0.3, hands=None):
    """Zonas de accesorios {lentes, collar}: máscara L, o ausente si la foto no muestra esa zona.
    Lentes: banda de los ojos (pose) dentro del rostro. Collar: base del cuello entre los hombros (pose); al
    generar solo se usa la piel visible (si la prenda cubre el cuello, no hay collar). Las manos nunca se tocan.
    Aretes y calzado se probaron y se retiraron: los aretes casi no se veían y el calzado fallaba 1-2 de cada 5."""
    out = {}
    face = _grow(np.isin(lab, _ids(id2label, ["Face"])), 2)
    hair = np.isin(lab, _ids(id2label, ["Hair"]))
    keep = hands if hands is not None else np.zeros(lab.shape, bool)
    to_l = lambda m: Image.fromarray((m * 255).astype(np.uint8), "L")  # noqa: E731
    if kps is None:
        return out
    if min(kps[5][2], kps[6][2]) >= min_score:
        c = ac.get("collar", {})
        (ax, ay, _), (bx, by, _) = kps[5], kps[6]
        sw = float(np.hypot(ax - bx, ay - by))  # ancho de hombros: escala del torso
        zone = _disc(lab.shape, (ax + bx) / 2, (ay + by) / 2 + sw * c.get("below", 0.08),
                     sw * c.get("half_width", 0.28), sw * c.get("half_height", 0.18))
        zone &= ~_grow(face | hair, 3) & ~keep
        if zone.any():
            out["collar"] = to_l(zone)
    if min(kps[1][2], kps[2][2]) < min_score:
        return out
    (lx, ly, _), (rx, ry, _) = kps[1], kps[2]
    d = float(np.hypot(lx - rx, ly - ry))  # distancia entre ojos: escala de la cabeza
    if d < ac.get("min_eye_px", 6):
        return out
    g = ac.get("lentes", {})
    band = _disc(lab.shape, (lx + rx) / 2, (ly + ry) / 2, d * g.get("half_width", 1.05), d * g.get("half_height", 0.42))
    if (band & face).any():
        out["lentes"] = to_l(band & (face | hair) & ~keep)
    return out


def restore_background(orig, gen, bg, max_diff=28):
    """El inpainting repinta un poco el fondo junto a la silueta («sombra» alrededor de las piernas). Donde el
    segmentador ve FONDO y el resultado casi no cambió (diferencia <= max_diff) se devuelve el original; las
    prendas que sobresalen de la silueta cambian mucho más y se conservan."""
    a, b = np.asarray(orig.convert("RGB")).astype(int), np.asarray(gen.convert("RGB")).astype(int)
    keep = bg & (np.abs(a - b).max(axis=2) <= max_diff)
    m = Image.fromarray((keep * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.5))
    return Image.composite(orig.convert("RGB"), gen.convert("RGB"), m)


def wrist_zone(shape, kps, min_score=0.3, back=0.10, radius=0.20):
    """Disco alrededor de cada muñeca (un poco hacia el codo), en fracciones del antebrazo. Con manga larga esa zona
    se repinta: una pulsera o reloj de la foto quedaba intacto junto al puño (el segmentador la marca como fondo)."""
    z = np.zeros(shape, bool)
    for e, w in ARMS:
        if min(kps[e][2], kps[w][2]) < min_score:
            continue
        (ex, ey, _), (wx, wy, _) = kps[e], kps[w]
        L = float(np.hypot(wx - ex, wy - ey))
        if L < 8:
            continue
        ux, uy = (wx - ex) / L, (wy - ey) / L
        z |= _disc(shape, wx - ux * L * back, wy - uy * L * back, L * radius, L * radius)
    return z


def clean_background(img, bg, grow=6):
    """Fondo sin la persona: se borra todo lo que no es fondo (con margen) y se rellena desde el fondo vecino
    (cv2.inpaint). Sirve para tapar el «halo» que deja la silueta anterior (p. ej. un jean ancho). None sin OpenCV."""
    try:
        import cv2
    except ImportError:
        return None
    hole = _grow(~bg, grow).astype(np.uint8) * 255
    out = cv2.inpaint(np.asarray(img.convert("RGB")), hole, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(out)


def paste_clean_background(gen, clean, bg_new, bg_old, grow=2):
    """Donde antes había persona y ahora el segmentador ve fondo (la prenda nueva es más angosta), se pone el
    fondo limpio: el modelo lo rellenaba con un tono más claro que el real."""
    fix = bg_new & ~bg_old
    m = Image.fromarray((_grow(fix, grow) & bg_new).astype(np.uint8) * 255, "L").filter(ImageFilter.GaussianBlur(1.2))
    return Image.composite(clean.convert("RGB"), gen.convert("RGB"), m)


def person_box(lab, id2label):
    """Caja COCO (x, y, ancho, alto) de la persona: todo lo que no es fondo."""
    ys, xs = np.nonzero(~np.isin(lab, _ids(id2label, ["Background"])))
    if len(xs) == 0:
        h, w = lab.shape
        return [0.0, 0.0, float(w), float(h)]
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def hands_from_pose(lab, id2label, kps, min_score=0.3, ahead=0.35, radius=0.6):
    """Píxeles de mano a partir del esqueleto: un círculo más allá de cada muñeca, en la dirección del antebrazo
    (codo -> muñeca), de radio proporcional a su largo. Solo píxeles de brazo: no conserva ropa vieja alrededor."""
    h, w = lab.shape
    yy, xx = np.mgrid[0:h, 0:w]
    hand = np.zeros((h, w), dtype=bool)
    for e, wr in ARMS:
        (ex, ey, es), (wx, wy, ws) = kps[e], kps[wr]
        if min(es, ws) < min_score:
            continue
        vx, vy = wx - ex, wy - ey
        length = max(float(np.hypot(vx, vy)), 8.0)
        cx, cy = wx + ahead * vx, wy + ahead * vy
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * length) ** 2
        # solo del lado de los dedos: con la mano en el cuello el círculo alcanzaba el hombro y el brazo, y esa piel
        # quedaba protegida (sin manga). Se admite un margen de 10 % del antebrazo antes de la muñeca.
        beyond = (xx - wx) * vx + (yy - wy) * vy >= -0.10 * length * length
        hand |= circle & beyond
    return hand & np.isin(lab, _ids(id2label, ["Left-arm", "Right-arm"]))


# Formato OpenPose-18 que espera ControlNet openpose: índice COCO de cada punto (cuello = media de hombros).
_OP_FROM_COCO = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
_OP_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
             (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_OP_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
              (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
              (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def openpose_image(kps, size, min_score=0.3, width=4):
    """Esqueleto en el formato de ControlNet openpose (fondo negro, extremidades de colores). No lleva textura de
    la foto: fija la pose sin copiar estampados ni marcas de agua (lo que hacía el control canny)."""
    from PIL import ImageDraw
    pts = []
    for c in _OP_FROM_COCO:
        if c is None:
            (ax, ay, a_s), (bx, by, b_s) = kps[5], kps[6]
            pts.append(((ax + bx) / 2, (ay + by) / 2, min(a_s, b_s)))
        else:
            pts.append(tuple(kps[c]))
    im = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(im)
    for i, (a, b) in enumerate(_OP_LIMBS):
        if pts[a][2] >= min_score and pts[b][2] >= min_score:
            col = tuple(int(v * 0.6) for v in _OP_COLORS[i])
            d.line([pts[a][:2], pts[b][:2]], fill=col, width=width * 2)
    for i, (x, y, s) in enumerate(pts):
        if s >= min_score:
            d.ellipse([x - width, y - width, x + width, y + width], fill=_OP_COLORS[i])
    return im


def skin(img):
    """Píxeles con color de piel (regla clásica en YCrCb, válida para tonos de piel claros y oscuros). Dentro de
    la zona de una mano se conserva solo la piel: la tela de una manga (p. ej. a rayas) se repinta, si no el
    modelo continuaba ese estampado por toda la prenda."""
    ycc = np.asarray(img.convert("YCbCr"), dtype=np.int16)
    cb, cr = ycc[..., 1], ycc[..., 2]
    return (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)


def not_background(img, sat=0.10, val=0.85):
    """Píxeles que no son fondo claro y sin color (blanco o gris de estudio). Se usa para que la zona protegida
    de una mano conserve solo la piel: el halo blanco que el segmentador pega al contorno se repinta con la prenda."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return ~((hsv[..., 1] < sat) & (hsv[..., 2] > val))


def face_box(lab, id2label, pad=0.15):
    """Caja (x0, y0, x1, y1) de la etiqueta Face con un margen; None si no hay rostro."""
    ys, xs = np.nonzero(np.isin(lab, _ids(id2label, ["Face"])))
    if len(xs) < 50:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    h, w = lab.shape
    return tuple(int(v) for v in (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)))


def count_faces_labels(lab, id2label, min_px=400):
    """Número de regiones de rostro separadas (componentes conexas de la etiqueta Face)."""
    from scipy import ndimage
    comp, n = ndimage.label(np.isin(lab, _ids(id2label, ["Face"])))
    return sum(int((comp == i).sum()) >= min_px for i in range(1, n + 1))


def face_ssim(orig, var, box):
    """Identidad conservada: SSIM de la región del rostro (None si no se detectó rostro)."""
    if box is None:
        return None
    return ssim(orig.crop(box), var.crop(box).resize(orig.crop(box).size))


# ------------------------------------------------------------------ simulados (sin modelos)
class MockSegmenter:
    id2label = MOCK_ID2LABEL

    def labels(self, img):
        h, w = img.size[1], img.size[0]
        lab = np.zeros((h, w), dtype=np.int64)
        lab[int(h * .05):int(h * .20), int(w * .35):int(w * .65)] = 2    # cabello
        lab[int(h * .20):int(h * .42), int(w * .38):int(w * .62)] = 11   # rostro
        lab[int(h * .48):, int(w * .25):int(w * .75)] = 4                # ropa superior
        lab[int(h * .48):, int(w * .12):int(w * .25)] = 14
        lab[int(h * .48):, int(w * .75):int(w * .88)] = 15
        return lab


def _tint(prompt):
    d = hashlib.sha256(prompt.encode()).digest()
    return d[0], d[1], d[2]


def mock_a(img, spec):
    """A simulado: tiñe toda la imagen (como img2img, también cambia el rostro)."""
    return Image.blend(img.convert("RGB"), Image.new("RGB", img.size, _tint(spec["prompt"])), 0.35), False


def mock_b(img, mask, spec, control=None):
    """B simulado: el mismo recorte y ampliación que el real, pintando un color dentro de la zona de la ropa."""
    return inpaint_region(lambda crop, m, size, c: (Image.new("RGB", size, _tint(spec["prompt"])), False),
                          img.convert("RGB"), mask, 768, control)


def mock_pose(img, box=None):
    """Esqueleto simulado coherente con MockSegmenter (17 puntos COCO: x, y, score); muñecas al final de los brazos."""
    w, h = img.size
    k = np.zeros((17, 3))
    k[0] = (.5 * w, .3 * h, 1)
    k[5], k[6] = (.70 * w, .5 * h, 1), (.30 * w, .5 * h, 1)       # hombros (izq./der. de la persona)
    k[7], k[8] = (.815 * w, .72 * h, 1), (.185 * w, .72 * h, 1)   # codos
    k[9], k[10] = (.815 * w, .90 * h, 1), (.185 * w, .90 * h, 1)  # muñecas
    return k


# ------------------------------------------------------------------ modelos reales (Colab)
class PoseEstimator:
    """ViTPose (transformers): 17 puntos del cuerpo (COCO) dentro de la caja de la persona."""

    def __init__(self, cfg, device="cuda"):
        import torch
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
        m = cfg["models"]["pose"]
        self.torch, self.device = torch, device
        self.proc = VitPoseImageProcessor.from_pretrained(m["repo"], revision=m["revision"])
        self.model = VitPoseForPoseEstimation.from_pretrained(m["repo"], revision=m["revision"]).to(device).eval()
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def detect(self, img, box):
        """box: [x, y, ancho, alto]. Devuelve un array (17, 3): x, y, confianza."""
        inp = self.proc(img, boxes=[[box]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        r = self.proc.post_process_pose_estimation(out, boxes=[[box]])[0][0]
        return np.concatenate([r["keypoints"].cpu().numpy(), r["scores"].cpu().numpy()[:, None]], axis=1)


class Segmenter:
    def __init__(self, cfg, device="cuda"):
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        m = cfg["models"]["segmenter"]
        self.torch, self.device = torch, device
        self.proc = SegformerImageProcessor.from_pretrained(m["repo"], revision=m["revision"])
        self.model = AutoModelForSemanticSegmentation.from_pretrained(m["repo"], revision=m["revision"]).to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.commit = getattr(self.model.config, "_commit_hash", None)

    def labels(self, img):
        t = self.torch
        with t.no_grad():
            logits = self.model(**self.proc(images=img, return_tensors="pt").to(self.device)).logits
            up = t.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy()


def _euler(pipe):
    from diffusers import EulerDiscreteScheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)
    return pipe


def load_inpaint(cfg, device="cuda", offload=False):
    import torch

    from .compat import neutralize_incompatible_torchao
    print(neutralize_incompatible_torchao())
    from diffusers import AutoPipelineForInpainting, ControlNetModel, StableDiffusionControlNetInpaintPipeline
    if not cfg["generation"]["safety_checker"]:
        raise ValueError("El safety checker debe permanecer activo (guardrail del proyecto).")

    def load(cls, entry, **extra):
        kw = dict(revision=entry["revision"], torch_dtype=torch.float16, use_safetensors=True, **extra)
        try:
            return cls.from_pretrained(entry["repo"], variant="fp16", **kw)
        except (OSError, ValueError):  # el repo no publica la variante fp16: se convierten los pesos completos
            return cls.from_pretrained(entry["repo"], **kw)
    m, cn = cfg["models"]["sd15_inpaint"], cfg["approaches"]["B"].get("controlnet")
    if cn and cn.get("enabled"):
        pipe = load(StableDiffusionControlNetInpaintPipeline, m, controlnet=load(ControlNetModel, cn))
    else:
        pipe = load(AutoPipelineForInpainting, m)
    pipe = _euler(pipe)
    # El filtro NSFW sigue activo, pero revisa la imagen FINAL (lo que ve el usuario), no el recorte ampliado
    # del torso: ese primer plano daba falsos positivos con prendas color piel. Ver gen_b y nsfw_check.
    checker = pipe.safety_checker
    pipe.safety_checker = None
    if offload:  # GPU de 4 GB (respaldo local): cada submodelo sube a la GPU solo mientras se usa
        pipe.enable_model_cpu_offload()
        checker = checker.to("cpu", torch.float32) if checker is not None else None
    else:
        pipe = pipe.to(device)
        checker = checker.to(device) if checker is not None else None
    pipe.vitrina_checker = (checker, pipe.feature_extractor)
    return pipe


def nsfw_check(pipe, img):
    """Safety checker de SD sobre una imagen completa. True = bloquear; None = no se pudo evaluar."""
    import torch
    checker, fe = getattr(pipe, "vitrina_checker", (pipe.safety_checker, pipe.feature_extractor))
    if checker is None:
        return None
    try:
        dev, dt = checker.device, checker.dtype
        clip_input = fe(images=img.convert("RGB"), return_tensors="pt").pixel_values.to(dev, dt)
        arr = (np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)[None]
        _, flags = checker(images=arr, clip_input=clip_input)
        return bool(flags[0])
    except Exception as e:  # noqa: BLE001  (la API del safety checker cambia entre versiones de diffusers)
        print("Aviso: no se pudo evaluar NSFW:", type(e).__name__, e)
        return None


def _nsfw(out):
    flags = getattr(out, "nsfw_content_detected", None)
    return bool(flags[0]) if flags else False


def gen_a(pipe, img, spec, cfg):
    import torch
    a, g = cfg["approaches"]["A"], cfg["generation"]
    out = pipe(prompt=spec["prompt"], negative_prompt=g["negative_prompt"], image=img, strength=a["strength"],
               num_inference_steps=a["steps"], guidance_scale=g["guidance_scale"],
               generator=torch.Generator("cuda").manual_seed(spec["seed"]))
    return out.images[0], _nsfw(out)


def work_box(mask, img_size, pad=0.12, min_side=192):
    """Caja de trabajo alrededor de la ropa (con margen de contexto). None si la máscara está vacía."""
    bb = mask.getbbox()
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    W, H = img_size
    px = max(int((x1 - x0) * pad), (min_side - (x1 - x0)) // 2)
    py = max(int((y1 - y0) * pad), (min_side - (y1 - y0)) // 2)
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


def work_size(box, side):
    """Tamaño de trabajo: el lado mayor de la caja a `side` px, ambos lados múltiplos de 8."""
    w, h = box[2] - box[0], box[3] - box[1]
    k = side / max(w, h)
    return max(8, round(w * k / 8) * 8), max(8, round(h * k / 8) * 8)


def blocked_card(size, text="Look bloqueado por el filtro de contenido"):
    """Lo que se muestra si el filtro NSFW bloquea un look tras los reintentos (en vez de una silueta negra)."""
    from PIL import ImageDraw
    im = Image.new("RGB", size, (226, 230, 238))
    d = ImageDraw.Draw(im)
    for i, line in enumerate([text, "Pide una nueva ronda para este look."]):
        l, t, r, b = d.textbbox((0, 0), line)
        d.text(((size[0] - (r - l)) // 2, size[1] // 2 - 14 + 18 * i), line, fill=(60, 70, 90))
    return im


def inpaint_region(run, img, mask, side, control=None, min_side=192):
    """Inpainting «solo la zona»: recorta la ropa, la amplía a `side` px (en una foto de cuerpo entero la ropa
    ocupa pocos píxeles y saldría sin detalle), la regenera con `run(recorte, máscara, (w, h)) -> (PIL, nsfw)` y
    la pega de vuelta. Fuera de la máscara se conserva el original: rostro, cabello y fondo intactos."""
    box = work_box(mask, img.size, min_side=min_side)
    if box is None:
        return img.copy(), False
    size = work_size(box, side)
    crop, mcrop = img.crop(box), mask.crop(box)
    ccrop = control.crop(box).resize(size, Image.Resampling.NEAREST) if control is not None else None
    res, nsfw = run(crop.resize(size, Image.Resampling.LANCZOS), mcrop.resize(size, Image.Resampling.NEAREST), size,
                    ccrop)
    full = img.copy()
    full.paste(res.resize(crop.size, Image.Resampling.LANCZOS), box[:2])
    return Image.composite(full, img, mask.filter(ImageFilter.GaussianBlur(2))), nsfw


def _has_pattern(text):
    import re
    return bool(re.search(r"strip|plaid|floral|print|pattern|check|dots", text, re.I))


def positive_for(spec, cfg):
    """Texto del look; en prendas lisas se pide explícitamente tela lisa (SD 1.5 tiende a inventar estampados)."""
    if spec.get("accessory"):
        return spec["prompt"]
    hint = cfg["generation"].get("plain_hint")
    text = spec["prompt"] if not hint or _has_pattern(spec["prompt"]) else f"{spec['prompt']}, {hint}"
    if spec.get("long_sleeves") and cfg["generation"].get("sleeves_hint"):
        text += ", " + cfg["generation"]["sleeves_hint"]  # sin esto a veces deja un brazo sin manga
    return text


def negative_for(spec, cfg):
    """Prompt negativo del look: excluye estampados salvo que la prenda pedida tenga uno (rayas, cuadros, floral…)."""
    g = cfg["generation"]
    neg = g["negative_prompt"]
    if spec.get("accessory"):
        acc_neg = cfg.get("accesorios", {}).get("negative_prompt", neg)
        return f"{spec['extra_negative']}, {acc_neg}" if spec.get("extra_negative") else acc_neg
    if g.get("pattern_negative") and not _has_pattern(spec["prompt"]):
        neg = g["pattern_negative"] + ", " + neg  # delante: CLIP corta a 77 tokens y al final se perdía
    return neg


def gen_b(pipe, img, mask, spec, cfg, control=None):
    import torch
    b, g = cfg["approaches"]["B"], cfg["generation"]

    def run(crop, mcrop, size, ccrop):
        kw = {}
        if getattr(pipe, "controlnet", None) is not None:  # sin bordes (p. ej. sin OpenCV): control vacío
            kw = dict(control_image=ccrop if ccrop is not None else Image.new("RGB", size),
                      controlnet_conditioning_scale=b["controlnet"]["conditioning_scale"],
                      control_guidance_end=b["controlnet"].get("guidance_end", 1.0))
        out = pipe(prompt=positive_for(spec, cfg), negative_prompt=negative_for(spec, cfg), image=crop, mask_image=mcrop,
                   strength=b["strength"], num_inference_steps=spec.get("steps") or b["steps"], guidance_scale=g["guidance_scale"],
                   width=size[0], height=size[1], generator=torch.Generator("cuda").manual_seed(spec["seed"]), **kw)
        return out.images[0], _nsfw(out)  # sin checker en el pipeline: aquí siempre False
    res, nsfw = inpaint_region(run, img, mask, b["work_size"], control, spec.get("min_side", 192))
    flag = nsfw_check(pipe, res)
    return res, bool(nsfw or flag)


def nsfw_input(pipe, img):
    """Guardrail de entrada: el safety checker de SD sobre la foto subida. None si no se pudo evaluar."""
    return nsfw_check(pipe, img)


def model_commit(pipe):
    """Commit de Hugging Face de los pesos cargados (para registrar revision: main)."""
    return getattr(getattr(pipe, "config", None), "_commit_hash", None) or getattr(pipe, "_commit_hash", None)
