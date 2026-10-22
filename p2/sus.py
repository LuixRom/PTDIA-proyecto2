"""System Usability Scale (SUS, Brooke 1996): 10 ítems de 1 (totalmente en desacuerdo) a 5 (totalmente de acuerdo)."""

ITEMS = [
    "Creo que me gustaría usar este sistema con frecuencia.",
    "Encontré el sistema innecesariamente complejo.",
    "Pensé que el sistema era fácil de usar.",
    "Creo que necesitaría el apoyo de una persona técnica para poder usar este sistema.",
    "Encontré que las diversas funciones del sistema estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en este sistema.",
    "Imagino que la mayoría de las personas aprenderían a usar este sistema muy rápidamente.",
    "Encontré el sistema muy engorroso de usar.",
    "Me sentí muy seguro/a usando el sistema.",
    "Necesité aprender muchas cosas antes de poder empezar a usar este sistema.",
]


def score(answers):
    """Puntuación SUS 0–100. Ítems impares (positivos): respuesta − 1; pares (negativos): 5 − respuesta; suma × 2.5."""
    if len(answers) != 10:
        raise ValueError(f"El SUS tiene 10 ítems; se recibieron {len(answers)}.")
    vals = []
    for i, a in enumerate(answers, start=1):
        if a in (None, ""):
            raise ValueError(f"Falta responder la pregunta {i}.")
        a = int(a)
        if not 1 <= a <= 5:
            raise ValueError(f"La pregunta {i} debe estar entre 1 y 5.")
        vals.append(a - 1 if i % 2 else 5 - a)
    return sum(vals) * 2.5
