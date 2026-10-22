# 04 — Diseño experimental

Objetivo: elegir el método generativo para el sistema final con evidencia comparable. **No se ejecuta nada sin aprobación del usuario.**

## 1. Preguntas
- P1. ¿Qué método produce 5 variaciones **distintas y útiles** que **conserven la identidad** de la persona?
- P2. ¿Qué coste en tiempo y VRAM tiene cada método en T4?
- P3. ¿Coinciden las métricas automáticas con el juicio humano?

## 2. Entradas
- **Imagen común obligatoria (I1):** un rostro frontal sintético o con licencia verificada, el mismo para todos los métodos.
- Campaña completa: + I2 (perfil o 3/4, otra iluminación) e I3 (otro grupo demográfico aparente), para observar sesgo y robustez.
- Preprocesamiento idéntico para todas: detectar exactamente 1 cara → recorte cuadrado con margen → 512×512 → RGB → eliminar EXIF. Se guarda el SHA-256 de la entrada.

## 3. Métodos (ver 03)
A0 clásico · A1 notebook (1 prompt, 5 seeds) · A2 img2img + 5 atributos · A3 A2 + ControlNet · (A4 InstructPix2Pix, solo si A2/A3 fallan en identidad).

Los 5 atributos son fijos y se usan iguales en A0, A2, A3 y A4:

| # | Atributo | Prompt (A2/A3) |
|---|---|---|
| V1 | Iluminación cálida (golden hour) | "portrait photo, warm golden hour lighting, same person" |
| V2 | Iluminación de estudio fría | "portrait photo, cool blue studio lighting, softbox" |
| V3 | Estilo acuarela | "watercolor painting portrait, soft brush strokes" |
| V4 | Expresión: sonrisa | "portrait photo, smiling, happy expression" |
| V5 | B/N de época | "1950s black and white film portrait, grain" |

Negative prompt común: "deformed, disfigured, extra eyes, blurry, lowres, watermark, text".

## 4. Condiciones controladas
- Hardware Colab T4. Se registran `torch`, `diffusers`, `transformers`, la revisión (commit) de cada modelo en HF y el driver/CUDA.
- fp16, `pipe.to("cuda")` sin offload, `EulerDiscreteScheduler`, 25 pasos, cfg 7.0, resolución 512×512.
- Strength por método: A1 0.40 (como el notebook); A2 0.50; A3 0.65 con `controlnet_conditioning_scale` 0.8.
  - Mini-barrido previo con I1: A2 ∈ {0.4, 0.5, 0.6} y A3 ∈ {0.55, 0.65, 0.75}. El valor se elige por la regla de la §8 y luego se congela.
- Semillas: `seed = 1000 + índice_variación` y generador en `cuda`. Para la campaña completa se usan 3 réplicas de semilla (1000, 2000, 3000) y así se mide la variabilidad por semilla.
- Calentamiento: 1 generación descartada antes de medir tiempos (la 1.ª tarda más: 30.7 s frente a 5 s, celda 36).

## 5. Registro (`experimentos/runs.jsonl`, una línea por imagen)
```json
{"run_id":"…","method":"A3","input_sha256":"…","variation":3,"attribute":"acuarela",
 "model":"stable-diffusion-v1-5/stable-diffusion-v1-5@<rev>","controlnet":"…@<rev>",
 "scheduler":"Euler","steps":25,"cfg":7.0,"strength":0.65,"seed":1003,
 "prompt":"…","negative_prompt":"…","width":512,"height":512,
 "time_s":6.2,"vram_peak_gb":4.1,"nsfw_flag":false,"output":"A3/I1/v3_s1003.png",
 "versions":{"torch":"…","diffusers":"…"}}
```

## 6. Métricas (prerregistradas, calculadas por script)

| Dimensión | Métrica | Dirección | Notas |
|---|---|---|---|
| Identidad | Coseno entre embeddings FaceNet (facenet-pytorch, VGGFace2) del original y la variación | ↑ | Umbral de "misma persona" a calibrar; referencia [E]: ~0.6 |
| Fidelidad estructural | LPIPS(original, variación) | ↓ moderado | SSIM solo como referencia (ver 02 §5) |
| Cumplimiento del atributo | CLIPScore(variación, texto del atributo) − CLIPScore(original, mismo texto) | ↑ | Mide el cambio inducido, no el absoluto |
| Diversidad del set | LPIPS medio por pares (10 pares) y distancia CLIP media por pares | ↑ | Aplica al set de 5 |
| Calidad / fallo | Cara detectada (sí/no), NSFW flag, imagen negra | — | Filtro duro |
| Coste | time_s, vram_peak_gb | ↓ | — |

Se reporta media ± desviación estándar por método, con las réplicas de semilla.

## 7. Evaluación humana ciega
- Interfaz Gradio de evaluación (independiente de la app final): muestra el original y luego **sets** de 5 imágenes con un identificador aleatorio. El método y el orden se ocultan; el mapeo se guarda en `blind_key.json` aparte.
- Cada evaluador (≥3 miembros del equipo; ideal ≥5 personas) da, por imagen: aceptar/rechazar, una escala de 1–5 de "sigue siendo la misma persona", 1–5 de "calidad visual" y un comentario opcional. Por set: ranking de los métodos.
- Acuerdo entre evaluadores: Fleiss κ (aceptar/rechazar) y Kendall W (rankings).
- [EXT] Juez auxiliar: un VLM según el doc 05. Sus puntuaciones se guardan **por separado** y nunca sustituyen a las humanas.

## 8. Detección de artefactos y regla de selección
1. **Filtro duro**: se descarta una variación si no se detecta cara, si salta el NSFW o la imagen sale negra, o si la identidad queda por debajo del umbral. Un método con más de 1/5 de fallos de media queda descalificado.
2. **Puntuación compuesta** por método (cada métrica normalizada min–max entre métodos): 0.35·identidad + 0.25·cumplimiento del atributo + 0.25·diversidad + 0.15·(1 − tiempo normalizado). [H] Los pesos se fijan **antes** de ver los resultados y se justifican por la rúbrica: identidad y diversidad pesan más.
3. **Preferencia humana**: tasa de aceptación y ranking medio.
4. Decisión: gana el método mejor en el punto 3. Si la diferencia con el segundo es menor que 1 desviación estándar entre evaluadores, se declara **empate** y se elige el más barato (punto 2, criterio de tiempo). Si las métricas (2) y los humanos (3) discrepan, se documenta como hallazgo y se prioriza el juicio humano.

## 9. Piloto (propuesto, pendiente de aprobación)
- I1 × {A0, A1, A2, A3} × 5 variaciones × 1 semilla = **20 imágenes** (+ barrido de strength: 6 × 5 = 30 más).
- [E] Tiempo en T4: ~5–8 min de generación + ~5 min de descargas (ControlNet ~1.4 GB; FaceNet, CLIP y LPIPS < 1 GB).
- Salida: `runs.jsonl`, una hoja de contacto PNG, una tabla de métricas y la sesión ciega con al menos 1 evaluador.
- Criterio de éxito del piloto: el pipeline produce exactamente 5 imágenes por método, el registro está completo y las métricas se calculan sin error. Así se decide si A3 justifica su coste frente a A2.

## 10. Campaña completa (después del piloto)
3 imágenes × 4 métodos × 5 variaciones × 3 semillas = 180 imágenes, [E] ~20–25 min en T4. Hay que guardar en Drive porque la sesión de Colab puede cortarse.


---

## Revisión 2 (2026-10-21)
Texto original conservado arriba. Detalle y fuentes en `10_verificaciones_y_plan_piloto.md`.
- El piloto se reduce a **1 imagen de entrada × {A1, A2} × 5 variaciones = 10 imágenes**, más 2 de control (calentamiento y repetición para comprobar el determinismo). Sustituye a las §2–§3 y §9 originales.
- Parámetros: 512², Euler, 20 pasos, strength 0.40, cfg 7.0, semillas 1000+k **pareadas** entre A1 y A2 (solo cambia el prompt). Se fijan en `config/pilot.yaml`.
- Métricas: SSIM, LPIPS, similitud CLIP (no identidad), CLIPΔ del atributo, diversidad por pares, caras detectadas con Haar, NSFW, imagen negra, tiempo y VRAM.
- Se retira FaceNet: la identidad se juzga en la evaluación humana ciega.
- Todo es **exploratorio**: sin inferencia estadística. La campaña de la §10 queda condicionada al resultado del piloto.
- Dataset: SFHQ-T2I, solo SDXL (doc 10 §2).
- La extensión con A2 a strength 0.55 **no** se incluye (decisión del usuario).
