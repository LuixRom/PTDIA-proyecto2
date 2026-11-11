"""Guardrails de Vitrina IA que no necesitan modelos (los de imagen están en p2/tryon.py).

Texto del usuario = frontera de confianza: se normaliza (minúsculas, sin tildes) y se compara por raíz de
palabra con la lista de config/vitrina.yaml (guardrails.blocked_terms), en español y en inglés.
"""
import re
import unicodedata

from PIL import Image, ImageDraw


class GuardrailError(ValueError):
    """Solicitud bloqueada por un guardrail. El mensaje se muestra al usuario tal cual."""


CATEGORY_MSG = {
    "sexual": "contenido sexual o que desviste a la persona",
    "cuerpo": "cambios del cuerpo, la piel o los rasgos",
    "menores": "referencias a menores de edad",
    "violencia": "violencia o armas",
    "marcas": "marcas registradas de terceros",
}


def normalize(text):
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return " ".join(t.lower().split())


def blocked(text, cfg):
    """Lista de (categoría, término) presentes en `text`. [] = permitido."""
    t = normalize(text)
    hits = []
    for cat, terms in cfg["guardrails"]["blocked_terms"].items():
        for term in terms:
            if re.search(r"(?<![a-z])" + re.escape(normalize(term)), t):
                hits.append((cat, term))
    return hits


def check_text(text, cfg, max_len=200):
    """Valida una instrucción escrita; lanza GuardrailError con un mensaje para el usuario."""
    t = " ".join((text or "").split())
    if not t:
        raise GuardrailError("Escribe qué prenda quieres probarte (por ejemplo: «una casaca verde de jean»).")
    if len(t) > max_len:
        raise GuardrailError(f"La instrucción es demasiado larga ({len(t)} caracteres; máximo {max_len}).")
    hits = blocked(t, cfg)
    if hits:
        cats = sorted({CATEGORY_MSG[c] for c, _ in hits})
        raise GuardrailError("Instrucción bloqueada por las reglas de uso del probador: " + "; ".join(cats) + ".")
    return t


def check_consent(own_photo, adult):
    if not own_photo or not adult:
        raise GuardrailError("Para continuar confirma que la foto es tuya (o tienes permiso) y que eres mayor de edad.")


def label(img, text="Simulación IA"):
    """Etiqueta visible en la esquina: la imagen es una simulación generada por IA, no una foto real."""
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((0, 0), text)
    w, h = r - l + 10, b - t + 8
    x, y = im.width - w - 6, im.height - h - 6
    d.rectangle([x, y, x + w, y + h], fill=(18, 53, 91))
    d.text((x + 5, y + 4 - t), text, fill=(255, 255, 255))
    return im


if __name__ == "__main__":  # autocomprobación rápida sin pytest
    from . import config
    c = config.load("config/vitrina.yaml")
    assert blocked("Quiero verme sin ropa", c) and blocked("make her thinner", c) and blocked("una casaca NIKE", c)
    assert not blocked("una casaca verde de jean", c) and not blocked("camino al trabajo", c)
    assert isinstance(label(Image.new("RGB", (64, 64))), Image.Image)
    print("guardrails ok")
