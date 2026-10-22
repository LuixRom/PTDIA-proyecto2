"""Registro persistente de decisiones: estado final en decisions.json + historial en events.jsonl."""
import json
import os
import threading
from pathlib import Path

from .schemas import Session, now_iso


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atómico en el mismo sistema de archivos


class DecisionStore:
    def __init__(self, root):
        self.root = Path(root)
        # Gradio atiende cada evento en un hilo: sin candado, dos "leer-modificar-guardar" simultáneos se pisan.
        self._lock = threading.RLock()

    def _dir(self, session_id):
        return self.root / session_id

    def create(self, session: Session):
        d = self._dir(session.session_id)
        with self._lock:
            if (d / "decisions.json").exists():
                raise FileExistsError(f"La sesión {session.session_id} ya existe; no se sobrescribe.")
            d.mkdir(parents=True, exist_ok=True)
            self._save(session)
            self._event(session.session_id, {"type": "created", "n": len(session.variations)})
        return session

    def load(self, session_id) -> Session:
        # Si el archivo está corrupto, pydantic/json lanza error y no se toca el archivo.
        return Session.model_validate_json((self._dir(session_id) / "decisions.json").read_text(encoding="utf-8"))

    def decide(self, session_id, var_id, decision, reason=None):
        """reason=None conserva el motivo ya registrado (un clic tardío no borra un motivo escrito después)."""
        if decision not in ("aceptada", "rechazada"):
            raise ValueError(f"Decisión inválida: {decision!r}")
        with self._lock:
            s = self.load(session_id)
            v = s.by_id(var_id)  # KeyError si el ID no existe
            previous = v.decision
            v.decision, v.decided_at = decision, now_iso()
            if reason is not None:
                v.reason = reason.strip()
            self._save(s)
            self._event(session_id, {"type": "decision", "id": var_id, "decision": decision,
                                     "previous": previous, "reason": v.reason})
        return s

    def summary_ready(self, session_id):
        return self.load(session_id).complete()

    def events(self, session_id):
        p = self._dir(session_id) / "events.jsonl"
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _save(self, s: Session):
        _atomic_write(self._dir(s.session_id) / "decisions.json", s.model_dump_json(indent=2))

    def _event(self, session_id, ev):
        ev = {"at": now_iso(), **ev}
        with open(self._dir(session_id) / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
