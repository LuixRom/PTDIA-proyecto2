# Piloto 2 — barrido de strength (0.55, 0.70), A1 vs A2 (resultados medidos)

Ejecución: Colab Tesla T4, diffusers 0.41.0, validación ok. Carpeta `results/pilot2_strength_a67c6d75_20261021-1749/`
(ZIP descargado SHA-256 `ecc87397…5889`, manifiesto íntegro). Solo cambió `strength`; imagen, semillas 1001–1005,
prompts, 20 pasos, Euler y cfg 7 iguales que el piloto 1 (strength 0.40, `piloto1_analisis.md`).

## Ejecución
- 23 generaciones (1 calentamiento + 2 × (10 + 1 determinismo)); determinismo idéntico en ambos strengths.
- Tiempo medio: 1.98 s/img (0.55), 2.61 s/img (0.70); VRAM pico 3.42 GB; 0 NSFW; 0 negras.
- Caras detectadas (Haar, orientativo): 1 en 19/20; A1@0.55 V2 = 2 (probable falso positivo).

## Métricas (sin LPIPS)
| Strength | Método | CLIPΔ medio | CLIP sim. medio (mín) | Diversidad CLIP | 1−SSIM |
|---|---|---|---|---|---|
| 0.40 | A1 | −0.007 | 0.957 (0.931) | 0.046 | 0.203 |
| 0.40 | A2 | −0.003 | 0.928 (0.914) | 0.037 | 0.196 |
| 0.55 | A1 | −0.013 | 0.816 (0.718) | 0.198 | 0.276 |
| 0.55 | A2 | +0.014 | 0.809 (0.680) | 0.191 | 0.270 |
| 0.70 | A1 | −0.013 | 0.623 (0.582) | 0.221 | 0.314 |
| 0.70 | A2 | +0.021 | 0.660 (0.597) | 0.340 | 0.333 |

CLIPΔ por atributo en A2 (0.55 / 0.70): luz cálida −0.012 / −0.002 · luz fría −0.002 / −0.018 ·
acuarela +0.055 / +0.070 · sonrisa +0.029 / +0.043 · B/N +0.001 / +0.010.

## Observación visual (hojas de contacto)
- 0.55: identidad mayormente conservada; sonrisa visible y algo de acuarela; B/N no aplicado; luz cálida/fría casi
  imperceptible. Cambios no pedidos: gafas, turbante/tocado, texto espurio en el fondo.
- 0.70: atributos más visibles (acuarela, sonrisa) pero **pérdida de identidad**: aparecen personas más jóvenes,
  de otro género y de otra etnia, también con el prompt genérico (A1). Contradice la regla ética de no alterar
  etnia/género/edad drástica → evidencia de sesgo del modelo hacia otros perfiles demográficos (para el informe).

## Lectura (exploratoria, n = 1 imagen, 1 semilla por variación)
- img2img solo ofrece un compromiso: más strength → más atributo y diversidad, menos parecido e identidad.
- 0.55 es el mejor compromiso de los tres; ninguno logra atributo claro + identidad estable.
- A2 > A1 en CLIPΔ en ambos strengths (A1 actúa como control): el condicionamiento por atributo funciona
  parcialmente; B/N e iluminación no se logran por prompt con SD 1.5 img2img a estos strengths.
- Siguiente opción técnica justificada por la evidencia: ControlNet (A3) para fijar la estructura facial y
  permitir más strength sin perder identidad.

## Evaluación humana ciega (strength 0.55) — 1 evaluador, 10 imágenes
Datos: `strength_0.55/human.jsonl` (respuestas) + `blind_key.json` → `strength_0.55/blind_results.csv`.
El clic extra tras terminar provocó un IndexError en la UI (sin pérdida de datos: 10 respuestas únicas);
corregido en `experiments/blind_eval.py`.

| Método | Aceptadas | "Misma persona" media (1–5) | Calidad media (1–5) |
|---|---|---|---|
| A1 | 3/5 | 3.2 | 4.2 |
| A2 | 4/5 | 3.0 | 4.0 |

- Rechazadas: A1-V3 (misma 3), A1-V4 (misma 1, CLIP 0.718), A2-V4 sonrisa (misma 1, CLIP 0.680).
- Spearman entre similitud CLIP y "misma persona": rho = 0.62, p = 0.056, n = 10 (tendencia, no concluyente).
- Lectura: sin diferencia concluyente A1 vs A2 con n tan pequeño; A2 aporta variaciones explicables por atributo.
  La similitud CLIP baja coincide con los juicios de "otra persona" → útil como aviso, nunca como identificación.

## Decisión (usuario, 2026-10-21)
Método para la app final: **A2 (5 atributos) con strength 0.55**. ControlNet (A3) queda como mejora documentada.

## Primera prueba de la app en Colab (2026-10-21) — valoración del usuario (no registrada por un fallo)
La sesión no quedó registrada: los clics de decisión no llegaban al registro en el visor de Colab y el motivo escrito
después de elegir se perdía. Corregido (las decisiones y motivos se leen al pulsar «Generar resumen»; candado contra
escrituras concurrentes en DecisionStore). Valoración comunicada por el usuario, A2 strength 0.55, imagen SDXL de prueba:
- V1 luz cálida, V2 luz fría, V3 acuarela: **aceptadas** ("se ve mejorado", "modo acuarela se ve bien").
- V4 sonrisa: **rechazada** — el personaje masculino pasa a ser mujer; sonrisa no natural (dientes deformados), corbata desunida.
- V5 B/N de época: **rechazada** — cambio de género del personaje; el resto bien (y no salió en B/N).
Lectura: el cambio de género no pedido reaparece a 0.55 en los atributos "sonrisa" y "B/N de época" → sesgo del modelo a
documentar en la sección ética; candidato a mitigación (prompt negativo de género / ControlNet) como trabajo futuro.

## Sesión 1 de la app en Colab (2026-10-22, `results/sesiones_app_20261022-0018/`, manifiesto íntegro)
- Decisiones registradas: V1 luz cálida, V2 luz fría, V3 acuarela **aceptadas**; V4 sonrisa y V5 B/N **rechazadas**
  (motivos del usuario: cambio de género no pedido; sonrisa y corbata deformadas). strength 0.55; 11 eventos.
- Resumen: Qwen2.5-1.5B **no superó** el verificador en 3 intentos ("V1, V2, V3: el resumen no indica explícitamente
  su estado") → se usó la plantilla determinista, marcada como respaldo y coherente (re-verificada).
- Diagnóstico: con decodificación greedy los 3 intentos eran idénticos; el texto del LLM no se guardaba.
  Cambios: formato fijo por variación + ejemplo (few-shot), reintento con los errores del verificador (bucle de
  revisión) y `raw_responses` en summary.json. Pendiente medir con nuevas sesiones.
