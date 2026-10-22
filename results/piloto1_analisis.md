# Piloto 1 — A1 vs A2 con strength 0.40 (resultados medidos)

Ejecución: Colab, Tesla T4 (15.6 GB), Python 3.13.15, torch 2.11.0+cu130, diffusers 0.41.0, transformers 5.20.1.
Carpeta: `results/pilot_A1_vs_A2_7bf36921_20261021-1735/` (ZIP descargado: SHA-256 `a4718d1b…7867`; manifiesto íntegro).

## Ejecución
- 12 generaciones: 1 calentamiento + A1 ×5 + A2 ×5 + 1 repetición de determinismo; semillas 1001–1005.
- ~1.5 s/imagen (calentamiento 4.3 s); VRAM pico 3.42 GB.
- Determinismo: misma semilla → archivo idéntico (`determinism_identical: true`).
- 0 NSFW, 0 imágenes negras, 1 cara detectada (Haar) en las 10.
- Validación: 40 comprobaciones; torchao 0.10.0 de Colab sin `FqnToConfig` → integración opcional desactivada (p2/compat.py).

## Métricas (sin LPIPS)
| Método | Diversidad CLIP (10 pares) | Diversidad 1−SSIM | CLIPΔ atributo (rango) | Similitud CLIP con original |
|---|---|---|---|---|
| A1 (prompt genérico) | 0.046 | 0.203 | −0.011 … −0.004 | 0.93–0.97 |
| A2 (5 atributos) | 0.037 | 0.196 | −0.011 … +0.005 | 0.91–0.94 |

## Lectura (exploratoria, n = 1 imagen)
- A1 y A2 son **visualmente casi idénticas** (hojas de contacto): no se ven acuarela, B/N ni cambios de iluminación.
- CLIPΔ ≈ 0 en los 5 atributos de A2: el atributo **no se aplicó** de forma medible.
- Causa probable: strength 0.40 × 20 pasos = 8 pasos de des-ruido efectivos; conserva la cara pero el prompt apenas influye (riesgo previsto en `planificacion/10`, riesgo 5).
- Decisión (usuario, 2026-10-21): **piloto 2** con strength 0.55 y 0.70, resto de parámetros, semillas y prompts sin cambios. La evaluación ciega se hará tras elegir strength.
