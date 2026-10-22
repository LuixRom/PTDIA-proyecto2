# 07 — Plan de implementación

Cada etapa termina con un hito verificable. Ninguna etapa E1+ empieza sin aprobación del usuario.

| Etapa | Contenido | Depende de | Hito verificable / criterio de aceptación |
|---|---|---|---|
| **E0** Aprobaciones | El usuario aprueba el piloto, el dataset y el juez visual | — | Decisiones registradas en este documento |
| **E1** Dataset y preprocesamiento | Elegir la fuente de rostros con licencia verificada (sintéticos preferentes). 20–30 imágenes de prueba. EDA (resolución, pose, iluminación, grupo aparente). `preprocess.py` | E0 | Tabla de licencias; notebook EDA con figuras; test: 0 caras → rechazo, 2 caras → rechazo, 1 cara → 512×512 sin EXIF |
| **E2** Piloto generativo | Scripts de 04 §9 en Colab: A0–A3 sobre I1 + barrido de strength, `runs.jsonl`, métricas | E1 | 20 (+30) imágenes; `runs.jsonl` completo (todas las claves); tabla de métricas; hoja de contacto |
| **E3** Evaluación ciega del piloto + juez | Interfaz ciega Gradio, ≥1–3 evaluadores, juez auxiliar (doc 05) | E2 | `blind_key.json` separado; `human.jsonl` y `judge_scores.jsonl`; decisión del método según 04 §8, **presentada al usuario** |
| **E4** Generador final | `generate.py` + `config.yaml` con el método ganador | E3 | Test `len(cinco_variaciones(...)) == 5`; misma seed → mismo hash de salida (dos ejecuciones); tiempo por sesión ≤ 60 s en T4 [objetivo] |
| **E5** Registro + agente explicador | `decisions.py`, `explainer.py`, prompt y verificador | E4 (puede empezar en paralelo con datos simulados) | Tests: el verificador detecta un ID intercambiado, un ID omitido, un ID inventado y un estado contrario; casos 0/5 y 5/5; ≥95 % de verificación al primer intento sobre 50 sesiones sintéticas |
| **E6** UI integrada | `ui.py`: carga → 5 tarjetas → decisiones → resumen → descarga; `run_colab.ipynb` | E4, E5 | Recorrido completo en Colab con `share=True`; botón de resumen deshabilitado hasta tener 5/5 decididas; el JSON descargado valida contra el esquema |
| **E7** Evaluación del sistema | Campaña completa (04 §10), SUS con ≥5 usuarios, 20 resúmenes revisados por humanos | E6 | Tablas finales de calidad, diversidad, coherencia y usabilidad, con IC o desviación estándar |
| **E8** Entregables | Informe ≤5 págs. (estructura basada en `main (3).pdf`), diagramas, ejemplos, video de demo, README de reproducción | E7 | PDF ≤5 págs.; video que muestra 5 variaciones, aceptar/rechazar y resumen; `08_matriz_de_requisitos.md` con todo "cumplido" |

Ruta crítica: E1 → E2 → E3 → E4 → E6 → E7 → E8. E5 va en paralelo a E2–E4.

## Riesgos

| Riesgo | Prob. | Impacto | Mitigación |
|---|---|---|---|
| SD 1.5 no conserva la identidad facial | Media-alta | Alto (rúbrica: modelo) | ControlNet; strength moderado; A4 o IP-Adapter como plan B; documentar como limitación |
| Safety checker con falsos positivos en caras | Media | Medio | Medir la tasa; regenerar con otra seed (registrado) en lugar de desactivarlo |
| Colab: desconexión, cuota de GPU, cambios de versiones | Alta | Medio | Versiones fijadas, salidas en Drive, revisiones de modelos fijadas, scripts idempotentes |
| SD + ControlNet + LLM 3B no caben a la vez en T4 | Media | Medio | Liberar SD antes del LLM o usar el 1.5B; medir VRAM pico en E2 |
| El LLM pequeño alucina o no respeta el JSON | Media | Alto (explicabilidad) | Verificador + reintento + fallback determinista marcado; greedy |
| Licencia o privacidad del dataset de rostros | Media | Alto (ética) | Sintéticos o con licencia verificada; consentimiento; no persistir por defecto |
| Disco local casi lleno (22 GB libres) | Alta si se ejecuta en local | Bajo (la ejecución principal es en Colab) | No descargar pesos en local |
| Coste económico | Baja | — | Todo gratuito (Colab free, modelos abiertos). Las APIs de pago quedan excluidas salvo aprobación |
| El informe supera 5 págs. | Media | Medio | Redactar desde la matriz 08 y priorizar figuras |

## Tareas en orden
1. Aprobar E0 (ver el resumen ejecutivo).
2. E1: elegir la fuente de rostros, verificar la licencia y preparar I1–I3.
3. E2: piloto en Colab.
4. E3: evaluación ciega + juez → decisión del método.
5. E5 (en paralelo desde el paso 3): esquema, verificador y tests.
6. E4 → E6 → E7 → E8.


---

## Revisión 2 (2026-10-21)
Texto original conservado arriba. Detalle y fuentes en `10_verificaciones_y_plan_piloto.md`.
Estado y nuevo orden:
1. Hecho: código del piloto y tests locales (88 en verde), `README.md`, `run_colab.ipynb`, `dist/proyecto2_colab.zip`.
2. Pendiente de aprobación: el usuario descarga las 10 imágenes SDXL de SFHQ-T2I y el CSV; luego se ejecuta `tools/select_images.py` y se completan `MANIFEST.csv` y `dataset.pilot_image`.
3. Pendiente de aprobación: posible instalación de `diffusers==0.40.0` y `lpips==0.1.4` en Colab (solo si el dry-run no cambia paquetes preinstalados) y descarga de ~3.6 GB.
4. Pendiente de aprobación aparte: **ejecutar el piloto**.
5. Después: evaluación ciega local → decisión A1/A2 → (opcional) ControlNet → LLM explicador real → UI final → evaluación (SUS) → informe.

No se crea repositorio público; la transferencia a Colab es por ZIP.
