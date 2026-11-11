"""Figuras del informe LaTeX (python informe/latex/make_figs.py)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
INK, BLUE, ORANGE, GREEN, RED, DASH = "#12355B", "#EEF2F8", "#FFF3DF", "#E9F5EA", "#FBE9E9", "#B23B3B"


def arquitectura():
    fig, ax = plt.subplots(figsize=(10, 4.55), dpi=220)
    ax.set_xlim(0, 100); ax.set_ylim(0, 45.5); ax.axis("off")

    def box(x, y, w, h, title, body="", fc=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", fc=fc, ec=INK, lw=0.9))
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="top", fontsize=6.8, weight="bold", color="#111")
        if body:
            ax.text(x + w / 2, y + h - 3.2, body, ha="center", va="top", fontsize=5.6, linespacing=1.22)

    def arrow(p, q, label="", color=INK, ls="-", off=(0, 0.8), fs=5.4):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color, linestyle=ls))
        if label:
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha="center", fontsize=fs,
                    color=color)

    X = [1, 21, 41, 61, 81]; W = 16; R1, R2, R3, H = 34, 18.5, 3, 10.5
    box(X[0], R1, W, H, "1. Cliente (web)", "sube su foto; elige la\nlínea, la ocasión o las\nprendas y cómo generar")
    box(X[1], R1, W, H, "2. Guardrails entrada", "una persona, mayor de\nedad, consentimiento;\ntérminos vetados; permiso\naparte para usar la API", RED)
    box(X[2], R1, W, H, "3. Preprocesamiento", "foto a 512×768 sin EXIF;\nmáscara de la ropa\n(SegFormer) y pose\n(ViTPose)")
    box(X[3], R1, W, H, "4. Generador", "Intermedio (local, gratis):\ninpainting + ControlNet\nBásico (API, de pago):\ngpt-image-2; lentes/collar", "#FFFFFF")
    box(X[4], R1, W, H, "5. Guardrails salida", "NSFW registrado; rostro\nconservado (similitud);\nexactamente 5: V1–V5", RED)
    box(X[4], R2, W, H, "6. Revisión (web)", "aceptar o rechazar por\nimagen o por paquete,\nmotivo; restaurar una\nconfiguración anterior", ORANGE)
    box(X[3], R2, W, H, "7. Registro", "decisiones (JSON),\nhistorial de parámetros\n(modelo, semilla, prompt),\neventos con hora", GREEN)
    box(X[2], R2, W, H, "8. Iteración", "regenera solo las\nrechazadas con otra\nsemilla y conserva\nlas aceptadas", ORANGE)
    box(X[2], R3, W, H, "9. Agente explicador", "2 LLM locales a elegir;\nprompt CONTEXTO, TAREA,\nRESTRICCIONES, FORMATO")
    box(X[3], R3, W, H, "10. Verificador", "sin LLM: IDs y estados\niguales al registro;\nsi falla, reintento\ncon los errores", RED)
    box(X[4], R3, W, H, "11. Salida", "resumen verificado;\nlooks guardados y\nenlace a las prendas\nde la tienda", GREEN)
    box(X[0], R3, 36, 26, "Métricas y monitoreo",
        "IA: ΔCLIP de la prenda pedida, identidad\nconservada (similitud del rostro), diversidad\nentre las 5 variaciones, bloqueos NSFW\n\n"
        "Explicabilidad: coherencia entre resumen\ny registro, número de reintentos\n\n"
        "Uso: tasa de aceptación, rondas hasta\naceptar, tiempo por decisión, SUS\n\n"
        "Negocio: costo por imagen (API)\no tiempo de GPU; looks guardados", "#F7F7F7")

    cy = R1 + H / 2
    for i in range(4):
        arrow((X[i] + W, cy), (X[i + 1], cy))
    arrow((X[4] + W / 2, R1), (X[4] + W / 2, R2 + H), "V1–V5", off=(2.8, 0))
    my = R2 + H / 2
    arrow((X[4], my), (X[3] + W, my), "decisión", off=(0, 0.8), fs=4.8)
    arrow((X[3], my), (X[2] + W, my))
    arrow((X[2] + W / 2 + 3, R2 + H), (X[3] + 3, R1), "nueva ronda", color=DASH, ls="--", off=(-4.2, -0.4))
    arrow((X[3] + 4, R2), (X[2] + W - 3, R3 + H), "registro", off=(3.4, 0.6))
    ly = R3 + 4
    arrow((X[2] + W, ly), (X[3], ly), "JSON", off=(0, 0.8), fs=4.8)
    arrow((X[3], ly + 4), (X[2] + W, ly + 4), "≤2", color=DASH, ls="--", off=(0, 0.6), fs=4.8)
    arrow((X[3] + W, ly), (X[4], ly), "ok", off=(0, 0.8), fs=4.8)
    fig.savefig(OUT / "arquitectura.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    arquitectura()
    print("figuras en", OUT)
