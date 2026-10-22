# 03 — Comparativa de alternativas generativas

Contexto: dominio de **rostros** con 5 variaciones por imagen; entorno Colab T4 (15 GB), sin presupuesto de API.
Marcas: **[M]** medido en el notebook · **[E]** estimación (documentación o experiencia, a verificar) · **[H]** hipótesis.

## 1. Candidatos

| ID | Técnica | Nivel (§2.3) | Descripción |
|---|---|---|---|
| A0 | Transformaciones clásicas (PIL/OpenCV: color grading, B/N, sepia, estilización, cambio de iluminación por curvas) | Bajo el básico (línea base) | Sirve de **control** para mostrar qué aporta el modelo generativo |
| A1 | SD 1.5 img2img, 1 prompt, 5 seeds (estado actual del notebook, Euler) | Básico | Reproduce la celda 44 |
| A2 | SD 1.5 img2img, **5 prompts de atributo** (iluminación cálida, estudio frío, acuarela, sonrisa, B/N de época), seed fija por atributo | Intermedio | Control explícito de estilo e iluminación |
| A3 | A2 + **ControlNet SD 1.5** (softedge/lineart o canny) | Intermedio (lo que nombra el enunciado) | Conserva contornos y pose con strength alto, lo que da más cambio de estilo sin perder composición |
| A4 | InstructPix2Pix (`timbrooks/instruct-pix2pix`) | Intermedio | Ediciones por instrucción ("make it night", "make him smile") con guidance de imagen |
| B1 | SDXL / SDXL-Turbo img2img | Básico/intermedio | Más calidad, más coste |
| B2 | APIs (DALL·E/gpt-image, Stability, Gemini) | Básico | De pago o con cuota; política de rostros restrictiva |
| B3 | Modelo propio (DCGAN/StyleGAN/pix2pix entrenado) | Avanzado | Entrenamiento no viable en el plazo y la GPU disponibles |
| — | IP-Adapter FaceID / InstantID | Intermedio | Preservación explícita de identidad; depende de insightface (modelos con **licencia no comercial** [E]) |

## 2. Matriz de 10 criterios

| Criterio | A0 | A1 | A2 | A3 | A4 | B1 SDXL | B2 API | B3 propio |
|---|---|---|---|---|---|---|---|---|
| 1. Calidad visual | Alta (sin artefactos) pero sin novedad [H] | Alta con Euler [M, celda 44]; mala con PNDM/512 [M] | Media-alta; riesgo de deformación facial con strength > 0.5 [E] | Alta; estructura estable [E] | Media; tiende a sobreeditar color [E] | Muy alta [E] | Muy alta [E] | Baja a 64–256 px [E] |
| 2. Fidelidad al original | Máxima | Alta: SSIM 0.53 [M] | Media; la identidad puede derivar [H] | Alta en composición; la identidad no está garantizada [H] | Media-alta [E] | Media | Variable; algunas APIs rechazan editar caras reales | Baja |
| 3. Diversidad útil entre las 5 | Media (reglas fijas) | **Muy baja**: casi idénticas [M] | **Alta**, por diseño semántico [H] | Alta [H] | Alta [H] | Alta | Alta | Baja-media |
| 4. Control (estilo, iluminación, composición) | Determinista, limitado | Bajo (solo strength) | Medio (prompt + strength) | **Alto** (prompt + mapa de control + escala) | Medio-alto (instrucción) | Medio | Bajo-medio | Bajo |
| 5. Coste | 0 | 0 (Colab gratis) | 0 | 0 | 0 | 0 en Colab; tiempo alto | **De pago** / cuota | 0 en dinero, alto en tiempo |
| 6. Tiempo por 5 variaciones (T4) | <1 s [E] | ~25–30 s [M: 5–6 s/img a 672×448] | ~25–35 s a 512² [E] | ~40–60 s [E: ControlNet añade ~30–50 %] | ~30–50 s [E] | ~2–4 min (Turbo: ~10 s, con peor fidelidad) [E] | ~10–60 s + red [E] | Horas de entrenamiento |
| 7. GPU/VRAM/RAM/disco | CPU | T4: ~3–4 GB VRAM fp16 [E]; 4 GB de disco [M] | Igual que A1 | +0.7 GB VRAM, +1.4 GB de disco (ControlNet fp16) [E] + preprocesador | ~3.5 GB VRAM, ~2.5 GB de disco [E] | ~8–10 GB VRAM, ~7 GB de disco [E] | Ninguno local | GPU dedicada durante horas |
| 8. Integración con la UI | Trivial | Fácil (función Python) | Fácil | Fácil (un pipeline más en Diffusers) | Fácil | Fácil, pero lenta para la demo | Fácil; requiere clave y red | Difícil |
| 9. Reproducibilidad | Total | Total: mismos bytes con la misma seed [M] | Total con seed y prompt fijos [E] | Total [E] | Total [E] | Total | **Baja** (modelos cambian sin aviso) | Media |
| 10. Riesgos/complejidad | No cumple "modelo generativo" como núcleo | No cumple la intención de "5 alternativas distintas"; nivel básico | Deriva de identidad; caras deformadas | Un componente más; el preprocesador puede fallar | Cambios no deseados de color global | VRAM/tiempo en T4, cortes de Colab | Coste, privacidad (subir caras a terceros), ToS | Fuera de plazo |

## 3. Valoración

- **Más prometedoras: A3 (img2img + ControlNet) y A2.** Son las que atacan el fallo principal medido, la falta de diversidad útil, con un control explícito que la rúbrica valora ("modelo generativo" 4 pts; nivel intermedio). A2 es el subconjunto barato de A3: si A3 no aporta en el piloto, A2 basta.
- **A4** es la alternativa si la identidad se degrada demasiado en A2/A3 o si las ediciones de expresión no funcionan solo con el prompt.
- **A0 y A1** entran **solo como líneas base** del experimento.
- **B1** solo si el piloto muestra que la calidad de SD 1.5 en caras es inaceptable. Coste: tiempo en T4.
- **B2** se descarta: subir caras a terceros añade un riesgo ético (§2.7), coste y baja reproducibilidad. Requeriría aprobación expresa.
- **B3** se descarta: el plazo y la GPU no permiten entrenar algo competitivo. Mencionarlo como trabajo futuro.

## 4. Incertidumbres principales
1. [H] Cuánto conserva la identidad SD 1.5 a strength 0.45–0.7 en rostros reales o sintéticos. **Es la incógnita crítica** y se mide con la similitud de embeddings faciales.
2. [E] Los tiempos de A2–A4 en T4 están sin medir.
3. [E] El safety checker de SD puede ennegrecer salidas con rostros (falsos positivos): hay que medir su tasa.
4. [E] Licencias: los pesos de SD 1.5 están bajo CreativeML OpenRAIL-M (uso permitido con restricciones de uso). Los modelos de insightface tienen licencia no comercial. Verificar antes de usar.


---

## Revisión 2 (2026-10-21)
Texto original conservado arriba. Detalle y fuentes en `10_verificaciones_y_plan_piloto.md`.
- Licencias verificadas (doc 10 §1.4):
  - SD 1.5: CreativeML Open RAIL-M; la licencia no reclama las imágenes generadas y obliga a respetar las restricciones de uso del Attachment A.
  - CLIP B/32: código MIT; su ficha restringe el uso a investigación y excluye el reconocimiento facial.
  - LPIPS: BSD-2.
- SDXL y FLUX.1-dev **no** se usan como generadores del sistema: solo son el origen de las imágenes del dataset SFHQ-T2I (generadas vía Stability API y fal.ai).
- ControlNet (A3) **aplazado**. El problema medido es la falta de diversidad, y A2 lo ataca sin descargas extra; A3 se reconsidera si A2 deforma la cara.
- InstructPix2Pix (A4) sigue como plan B, sin cambios.
- Qwen2.5-3B descartado (licencia "other"). El LLM explicador, cuando se use, será Qwen2.5-1.5B (Apache-2.0); en el piloto no se usa ninguno.
