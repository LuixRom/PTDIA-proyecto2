"""Evaluación humana ciega en Windows sobre la carpeta blind/ exportada por el piloto.

Uso:  python -m experiments.blind_eval <carpeta_resultados> --evaluador NOMBRE
No abre blind_key.json: el método queda oculto hasta el análisis.
"""
import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from p2.schemas import DecisionValue, now_iso


class Rating(BaseModel):
    evaluator: str = Field(min_length=1)
    image_id: str = Field(pattern=r"^B\d{2}$")
    decision: DecisionValue
    same_person: int = Field(ge=1, le=5)
    quality: int = Field(ge=1, le=5)
    artifacts: str = ""
    at: str = Field(default_factory=now_iso)


def blind_ids(blind_dir):
    return sorted(p.stem for p in Path(blind_dir).glob("B*.png"))


def record(results_dir, rating: Rating):
    with open(Path(results_dir) / "human.jsonl", "a", encoding="utf-8") as f:
        f.write(rating.model_dump_json() + "\n")


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--evaluador", required=True)
    a = ap.parse_args()
    blind = Path(a.results_dir) / "blind"
    ids = blind_ids(blind)
    with gr.Blocks(title="Evaluación ciega") as demo:
        gr.Markdown("Compara cada imagen con el original. No se muestra qué método la produjo.")
        idx = gr.State(0)
        with gr.Row():
            gr.Image(str(blind / "original.png"), label="Original", interactive=False)
            cur = gr.Image(str(blind / f"{ids[0]}.png"), label=ids[0], interactive=False)
        dec = gr.Radio(["aceptada", "rechazada"], label="Decisión")
        same = gr.Slider(1, 5, step=1, value=3, label="¿Sigue siendo la misma persona? (1–5)")
        qual = gr.Slider(1, 5, step=1, value=3, label="Calidad visual (1–5)")
        art = gr.Textbox(label="Artefactos o cambios no pedidos")
        btn = gr.Button("Guardar y siguiente")
        msg = gr.Markdown()

        def save(i, d, s, q, t):
            if i >= len(ids):  # clic extra tras terminar: no guardar nada ni fallar
                return i, gr.update(), "Evaluación ya completa: no se guardó nada más."
            if d is None:
                raise gr.Error("Elige aceptar o rechazar.")
            record(a.results_dir, Rating(evaluator=a.evaluador, image_id=ids[i], decision=d,
                                         same_person=int(s), quality=int(q), artifacts=t))
            i += 1
            if i >= len(ids):
                return i, gr.update(), "Evaluación completa. Gracias."
            return i, gr.update(value=str(blind / f"{ids[i]}.png"), label=ids[i]), f"{i}/{len(ids)} guardadas"
        btn.click(save, [idx, dec, same, qual, art], [idx, cur, msg])
    demo.launch()


if __name__ == "__main__":
    main()
