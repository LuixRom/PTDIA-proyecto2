"""BD temporal (SQLite, biblioteca estándar) para la prueba con participantes.

Guarda las respuestas del agente explicador (resumen, verificación, intentos y textos del LLM) y los cuestionarios SUS.
Vive dentro de la carpeta de sesiones de Colab (/content/sesiones): la celda 10 la exporta en el ZIP y la celda 11 la
borra junto con las sesiones. Colab también borra /content al cerrar el entorno. No contiene la foto original.
"""
import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

from .schemas import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS resumenes (
  id INTEGER PRIMARY KEY, session_id TEXT, participante TEXT, llm TEXT, verificado INTEGER, respaldo INTEGER,
  intentos INTEGER, resumen TEXT, aceptadas TEXT, rechazadas TEXT, respuestas_llm TEXT, errores TEXT, creado TEXT);
CREATE TABLE IF NOT EXISTS sus (
  id INTEGER PRIMARY KEY, participante TEXT, session_id TEXT, respuestas TEXT, puntuacion REAL, creado TEXT);
"""


class TempDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # gradio atiende a varios participantes en hilos distintos
        with self._lock, closing(sqlite3.connect(self.path)) as c:
            c.executescript(SCHEMA)

    def _exec(self, sql, args):
        with self._lock, closing(sqlite3.connect(self.path)) as c:
            c.execute(sql, args)
            c.commit()

    def add_summary(self, session_id, participant, res):
        s = res.summary
        self._exec("INSERT INTO resumenes (session_id, participante, llm, verificado, respaldo, intentos, resumen, "
                   "aceptadas, rechazadas, respuestas_llm, errores, creado) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (session_id, participant, res.llm, int(res.verified), int(res.fallback), res.attempts, s.resumen,
                    json.dumps(s.aceptadas), json.dumps(s.rechazadas),
                    json.dumps(res.raw_responses, ensure_ascii=False), json.dumps(res.errors, ensure_ascii=False),
                    now_iso()))

    def add_sus(self, participant, session_id, answers, score):
        self._exec("INSERT INTO sus (participante, session_id, respuestas, puntuacion, creado) VALUES (?,?,?,?,?)",
                   (participant, session_id, json.dumps(answers), score, now_iso()))

    def rows(self, table):
        if table not in ("resumenes", "sus"):
            raise ValueError(f"tabla desconocida: {table}")
        with self._lock, closing(sqlite3.connect(self.path)) as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute(f"SELECT * FROM {table} ORDER BY id")]
