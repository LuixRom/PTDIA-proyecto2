"""Verificador determinista: ¿el resumen refleja exactamente las decisiones registradas?

Independiente del LLM: solo compara datos. Devuelve una lista de errores ([] = coherente).
"""
import re

from .schemas import Session

_ID = re.compile(r"\bV\d+\b")
_SENT = re.compile(r"[.;\n]+")
# Palabra de estado; el grupo 'neg' detecta "no fue aceptada", "no se rechazó", etc.
_STATE = re.compile(r"(?P<neg>\bno\s+(?:(?:fue|fueron|ha sido|han sido|se)\s+)?)?(?P<w>acept|rechaz)", re.IGNORECASE)


def _state_of(word, negated):
    s = "aceptada" if word.lower().startswith("acept") else "rechazada"
    if negated:
        s = "rechazada" if s == "aceptada" else "aceptada"
    return s


def _stated(sentence, pos):
    """Estado más cercano antes del ID en la frase; si no hay, el primero después."""
    before = [m for m in _STATE.finditer(sentence) if m.start() < pos]
    after = [m for m in _STATE.finditer(sentence) if m.start() > pos]
    m = before[-1] if before else (after[0] if after else None)
    return None if m is None else _state_of(m.group("w"), bool(m.group("neg")))


def verificar(summary: dict, session: Session) -> list[str]:
    errors = []
    acc, rej = summary.get("aceptadas"), summary.get("rechazadas")
    text = summary.get("resumen")
    if not isinstance(acc, list) or not isinstance(rej, list) or not isinstance(text, str) or not text.strip():
        return ["Formato inválido: se esperan 'aceptadas' (lista), 'rechazadas' (lista) y 'resumen' (texto)."]
    if not session.complete():
        errors.append("La sesión tiene variaciones sin decidir.")

    true_acc, true_rej = set(session.accepted()), set(session.rejected())
    valid = {v.id for v in session.variations}
    if set(acc) != true_acc:
        errors.append(f"'aceptadas' = {sorted(set(acc))}, registro = {sorted(true_acc)}.")
    if set(rej) != true_rej:
        errors.append(f"'rechazadas' = {sorted(set(rej))}, registro = {sorted(true_rej)}.")
    if set(acc) & set(rej):
        errors.append(f"IDs en ambas listas: {sorted(set(acc) & set(rej))}.")
    if len(acc) != len(set(acc)) or len(rej) != len(set(rej)):
        errors.append("IDs repetidos en las listas.")

    invented = sorted(set(_ID.findall(text)) - valid)
    if invented:
        errors.append(f"El resumen menciona IDs inexistentes: {invented}.")

    truth = {v.id: v.decision for v in session.variations}
    mentioned = set()
    for sentence in _SENT.split(text):
        for m in _ID.finditer(sentence):
            vid = m.group()
            if vid not in valid:
                continue
            mentioned.add(vid)
            st = _stated(sentence, m.start())
            if st is None:
                errors.append(f"{vid}: el resumen no indica explícitamente su estado.")
            elif st != truth[vid]:
                errors.append(f"{vid}: el resumen dice '{st}' pero el registro dice '{truth[vid]}'.")
    for vid in sorted(valid - mentioned):
        errors.append(f"{vid}: no se menciona en el resumen.")
    return errors
