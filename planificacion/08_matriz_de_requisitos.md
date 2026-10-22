# 08 — Matriz de requisitos

Fuente: `doc-principal/EvaluaciónProyecto2 (1) (1).pdf`. "Estado actual" se refiere al trabajo existente (`Untitled1.ipynb`) a fecha 2026-10-21.

| # | Requisito | Fuente | Cómo se verificará | Estado actual |
|---|---|---|---|---|
| R1 | Procesar una imagen de entrada **proporcionada por el usuario** | §1 p1; §2.3 p2; §2.2 | Subida en la UI + validación (E1/E6) | **Parcial**: `files.upload()` en Colab, sin validación |
| R2 | Generar **exactamente 5** variaciones por imagen | §2.2 p2; §2.5 p3; §3.3 p4 | `assert len == 5` + test; video | **Parcial**: 5 imágenes generadas, pero casi idénticas (sin variedad útil) |
| R3 | Modelo generativo según el nivel elegido, con justificación | §2.3 p2; §3.3 p4 | Informe + 03/04 | **Parcial**: SD 1.5 img2img (nivel básico); falta el control (intermedio) y la justificación |
| R4 | Analizar el comportamiento del modelo: calidad, diversidad, limitaciones | §3.3 p4 | Métricas + análisis (E2/E7) | **Parcial**: barrido de strength y SSIM; sin análisis escrito |
| R5 | GUI que muestre las 5 variaciones | §2.5 p3; §3.4 p4 | Recorrido en Gradio (E6) | **Parcial**: ipywidgets en el notebook |
| R6 | Aceptar o rechazar **cada** opción | §2.2 p2; §3.4 p4 | Botón de resumen bloqueado hasta 5/5 decididas | **Parcial**: los widgets existen pero perdieron el estado (celdas 47–51) |
| R7 | Registro **estructurado** de decisiones, utilizable por el agente | §2.2; §3.4 p4 | Esquema pydantic + JSON validado | **Parcial / no trazable**: CSV de origen desconocido (celda 52) |
| R8 | Agente LLM que genere un resumen que diga **explícitamente** cuáles se aceptaron y cuáles se rechazaron | §2.2; §2.5; §3.5 p4 | Verificador determinista + tests | **Pendiente** (Qwen2-VL solo describe imágenes) |
| R9 | Resumen **coherente** con las decisiones registradas | §2.2; §3.5 | % que pasa el verificador; revisión humana | **Pendiente** |
| R10 | Dataset seleccionado y justificado (rostros sugeridos) | §2.4 p2; §3.2 p3 | Tabla de licencias + justificación | **Pendiente** (un paisaje sin licencia) |
| R11 | Análisis exploratorio del dataset | §3.2 p3 | Notebook EDA | **Pendiente** |
| R12 | Preprocesamiento (normalización, redimensionamiento, filtrado) | §3.2 p3 | Tests de `preprocess.py` | **Parcial**: solo `thumbnail(512)` |
| R13 | Métricas de calidad visual | §2.6 p3 | 04 §6 | **Parcial**: SSIM |
| R14 | Métricas de diversidad | §2.6 | LPIPS/CLIP por pares | **Parcial**: 1−SSIM |
| R15 | Métrica de coherencia decisiones↔resumen | §2.6 | Tasa del verificador | **Pendiente** |
| R16 | Métrica de usabilidad | §2.6 | SUS + tiempos | **Pendiente** |
| R17 | Análisis crítico, limitaciones y mejoras | §2.6; §3.5 | Sección del informe | **Pendiente** |
| R18 | Reflexión ética: identidad, sesgos, impacto real | §2.7 p3 | Sección del informe + medidas de 06 §4 | **Pendiente** |
| R19 | Definición del problema, contexto, objetivos y restricciones | §3.1 p3 | Informe §1 | **Pendiente** |
| R20 | Diagrama del flujo de información y decisiones | §3.1 p3; §4 | Mermaid / figura (06 §1) | **Parcial**: borrador en 06 |
| R21 | Código organizado, documentado y reproducible, con instrucciones | §4 p4 | `run_colab.ipynb` desde cero + README; versiones fijadas | **Pendiente** (notebook con celdas rotas o duplicadas y `pip -U`) |
| R22 | Informe técnico de **≤5 páginas** con todas las etapas | §4 p4 | Contar páginas del PDF | **Pendiente** |
| R23 | Elementos visuales: arquitectura, flujo, ejemplos, evidencia de interacción | §4 p4 | Figuras en el informe | **Pendiente** |
| R24 | Demo en vivo o video: 5 variaciones, aceptar/rechazar, resumen automático | §4 p5 | Video grabado | **Pendiente** |
| R25 | Equipo de 3–5 con reparto equilibrado | §2.1 p1–2 | Sección de roles en el informe | **No verificable** desde los archivos |
| R26 | Decisiones técnicas justificadas en el documento | §2.1 p2 | Revisión del informe frente a 03/04 | **Pendiente** |

Correspondencia con la rúbrica (p5):
- Diseño (4): R19, R20, R26.
- Modelo generativo (4): R2–R4.
- Interacción H-IA (4): R5–R7.
- Explicabilidad (3): R8, R9, R15.
- Evaluación (3): R13–R17.
- Ética y documentación (2): R18, R21–R24.


---

## Revisión 2 (2026-10-21)
Texto original conservado arriba. Detalle y fuentes en `10_verificaciones_y_plan_piloto.md`.
| # | Estado actualizado | Evidencia |
|---|---|---|
| R2 | Parcial → **mecanismo implementado** (`cinco_variaciones` exige 5; tests) | `tests/test_generate_mock.py`, `test_pilot_mock.py` |
| R5–R7 | Parcial → **implementado con mock** (UI Gradio, registro JSON + eventos) | `p2/ui.py`, `p2/decisions.py`, tests |
| R8–R9 | Pendiente → **parcial**: explicador con stub + verificador determinista; falta el LLM real | `p2/explainer.py`, `p2/verifier.py` |
| R10 | Pendiente → **parcial**: fuente y licencia verificadas; descarga pendiente de aprobación | doc 10 §1–§2 |
| R12 | Parcial → **implementado** (validación, 512², EXIF, sha256) | `p2/preprocess.py` |
| R13–R14 | Parcial → **implementado** (pendiente de ejecutar en Colab) | `p2/metrics.py` |
| R15 | Pendiente → **implementado** (tasa del verificador) | `p2/verifier.py` |
| R21 | Pendiente → **parcial**: notebook reproducible, configuración centralizada, versiones fijadas, tests | `run_colab.ipynb`, `config/`, `requirements/` |
