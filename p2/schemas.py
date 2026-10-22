"""Esquemas de datos validados (pydantic v2)."""
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

NUM_VARIATIONS = 5
VARIATION_IDS = [f"V{k}" for k in range(1, NUM_VARIATIONS + 1)]
DecisionValue = Literal["aceptada", "rechazada"]


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Variation(BaseModel):
    id: str = Field(pattern=r"^V[1-5]$")
    method: str
    attribute: str
    prompt: str = Field(min_length=1)
    seed: int
    params: dict = {}
    image_file: str = ""
    metrics: dict = {}
    decision: Optional[DecisionValue] = None
    reason: str = ""
    decided_at: Optional[str] = None


class Session(BaseModel):
    session_id: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(default_factory=now_iso)
    participant: str = Field(default="", max_length=40)  # alias anónimo (prueba con participantes)
    variations: list[Variation]

    @field_validator("variations")
    @classmethod
    def exactly_five_unique(cls, v):
        if len(v) != NUM_VARIATIONS:
            raise ValueError(f"Se requieren exactamente {NUM_VARIATIONS} variaciones, hay {len(v)}.")
        if sorted(x.id for x in v) != VARIATION_IDS:
            raise ValueError(f"Los IDs deben ser {VARIATION_IDS} sin repetir.")
        return v

    def by_id(self, var_id):
        for v in self.variations:
            if v.id == var_id:
                return v
        raise KeyError(var_id)

    def complete(self):
        return all(v.decision is not None for v in self.variations)

    def accepted(self):
        return sorted(v.id for v in self.variations if v.decision == "aceptada")

    def rejected(self):
        return sorted(v.id for v in self.variations if v.decision == "rechazada")


class Summary(BaseModel):
    aceptadas: list[str]
    rechazadas: list[str]
    resumen: str = Field(min_length=1)


class ExplainResult(BaseModel):
    summary: Summary
    verified: bool
    errors: list[str] = []
    attempts: int
    fallback: bool
    llm: str
    raw_responses: list[str] = []  # textos del LLM tal cual (análisis de la explicabilidad)


class RunRecord(BaseModel):
    """Una línea de runs.jsonl: una imagen generada en el piloto."""
    run_id: str
    method: str
    variation: str = Field(pattern=r"^V[1-5]$")
    attribute: str
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    strength: float
    guidance_scale: float
    scheduler: str
    width: int
    height: int
    model: str
    input_sha256: str
    output: str
    output_sha256: str
    time_s: float
    vram_peak_gb: Optional[float] = None
    nsfw: Optional[bool] = None
    role: Literal["main", "warmup", "determinism"] = "main"

    @model_validator(mode="after")
    def positive(self):
        if self.steps <= 0 or not 0 < self.strength <= 1:
            raise ValueError("steps > 0 y 0 < strength <= 1")
        return self
