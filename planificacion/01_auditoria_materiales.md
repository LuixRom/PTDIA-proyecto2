# 01 — Auditoría de materiales

Fecha: 2026-10-21. Alcance: `doc-principal/`, `Material/` y `Untitled1.ipynb` (el notebook se analiza en `02_auditoria_notebook.md`).

Leyenda de procedencia usada en todos los documentos:
- **[REQ]** requisito explícito del enunciado oficial.
- **[MAT]** sugerencia o fundamento tomado del material del curso.
- **[HIP]** hipótesis o propuesta propia (requiere verificación).
- **[EXT]** fuente externa al curso (paper/documentación), no incluida en `Material/`.

## 1. Inventario (24 archivos)

| Archivo | Formato | Tamaño | Págs./celdas | Relevancia | Cómo se revisó |
|---|---|---|---|---|---|
| `doc-principal/EvaluaciónProyecto2 (1) (1).pdf` | PDF | 0.2 MB | 5 págs. | **Enunciado oficial** | Leído completo (texto + imagen de página) |
| `Untitled1.ipynb` | Notebook Colab | 31 MB | 67 celdas | **Trabajo previo** | Código y salidas completas; imágenes embebidas extraídas e inspeccionadas |
| `Material/Session10.0_GenerativeModels.pdf` | PDF (slides) | 2.1 MB | 19 | Alta | Texto completo (pdftotext) |
| `Material/Session10.1_VAE.pdf` | PDF | 2.8 MB | 52 | Media | Texto completo |
| `Material/Sem10_Prompting&OrquestaciónDeAgentes.pdf` | PDF | 2.0 MB | 73 | **Alta** (agente) | Texto completo |
| `Material/Sem9_ModelosFundacionales.pdf` | PDF | 10.3 MB | 92 | **Alta** (ética, prompting) | Texto completo |
| `Material/Session7.0_Contrastive Learning.pdf` | PDF | 3.3 MB | 62 | Media (CLIP p58–61) | Págs. 1–2 + búsqueda de palabras clave |
| `Material/Session6.0_Segmentation.pdf` | PDF | 4.4 MB | 46 | Baja (U-Net p19–20) | Págs. 1–2 + búsqueda |
| `Material/Session9.0_Teacher-student.pdf` | PDF | 3.3 MB | 42 | Baja (DINO) | Págs. 1–2 + búsqueda |
| `Material/main (3).pdf` | PDF | 1.7 MB | 9 | Media (plantilla de informe) | Texto completo |
| `Material/2106.09685v2.pdf` | PDF (paper LoRA) | 1.5 MB | 26 | Baja | Título y abstract |
| `Material/main.ipynb` | Notebook | <0.1 MB | 10 | Media (salida JSON) | Código completo |
| `Material/lora_eval_official.ipynb` | Notebook | <0.1 MB | 7 | Media (validación de esquema) | Código completo |
| `Material/Sem10_Learning.ipynb` | Notebook | <0.1 MB | 8 | Baja | Código completo |
| `Material/Session1.0_LinearModel.pdf` … `Session9.1_JEPA.pdf` (10 restantes) | PDF | 2.3–6.3 MB | 18–126 | Baja (teoría general) | Págs. 1–2 + búsqueda de palabras clave |

Limitaciones de la revisión:
- Los PDF se leyeron como **texto extraído**; los contenidos que solo existen como figura no se inspeccionaron.
- Los 13 decks de teoría general solo se revisaron en sus primeras páginas y por búsqueda de palabras clave; un tema con otro nombre podría haberse pasado por alto.
- `lora_eval_official.ipynb` importa `dataset_generator.py` y `evaluation.py`, que **no están** en la carpeta: no es ejecutable tal cual.
- `Sem9_ModelosFundacionales.pdf` p52 anuncia una sesión de "IA Creativa (Visual)" que **no está en `Material/`**. Por eso no hay material del curso sobre difusión aplicada, img2img ni ControlNet.

## 2. Resumen del enunciado

Proyecto #2 "Sistema Generativo Interactivo para Variaciones Visuales con Feedback Humano y Explicación mediante Agentes de IA". Curso Planificación y Toma de Decisiones en IA, Dra. Aurea Soriano-Vargas.

| Tema | Contenido | Fuente |
|---|---|---|
| Objetivo | Generar variaciones de una imagen subida por el usuario, que este las acepte o rechace, y que un agente LLM explique la selección | §1, p.1 |
| Equipo | 3–5 estudiantes; decisiones técnicas justificadas en el documento | §2.1, p.1–2 |
| Núcleo obligatorio | **Exactamente 5** variaciones; GUI; aceptar/rechazar **cada** opción; registro **estructurado**; resumen LLM que diga explícitamente cuáles se aceptaron y cuáles se rechazaron, coherente con el registro | §2.2 p.2; §2.5 p.3; §3.3–3.5 p.4 |
| Niveles | Básico: preentrenado/API. Intermedio: img2img, ControlNet, control de estilo, iluminación o composición. Avanzado: modelo propio (GAN, difusión, pix2pix, CycleGAN, ControlNet) | §2.3 p.2 |
| Dataset | Elegir y justificar; se **sugieren rostros**; uso responsable; EDA; preprocesamiento | §2.4 p.2; §3.2 p.3 |
| Evaluación | Métricas de calidad visual, diversidad, coherencia decisiones↔resumen y usabilidad; análisis crítico | §2.6 p.3 |
| Ética | Manipulación de identidad, sesgos, impacto real | §2.7 p.3 |
| Diseño | Problema, contexto, objetivos y restricciones; diagrama de flujo de información y decisiones | §3.1 p.3 |
| Entregables | Código organizado y reproducible; informe de **≤5 páginas**; diagramas, ejemplos e interacción; demo en vivo o video | §4 p.4–5 |
| Rúbrica (20 pts) | Diseño 4 · Modelo generativo 4 · Interacción H-IA 4 · Explicabilidad 3 · Evaluación 3 · Ética+documentación 2 | p.5, Cuadro 1 |

Lectura crítica [HIP]:
- El enunciado no obliga a que el feedback **reentrene** ni **regenere**. Basta con registrarlo y explicarlo.
- "Dataset" es ambiguo para un sistema de inferencia. Se interpreta como el conjunto de imágenes de prueba/evaluación que hay que justificar, explorar (EDA) y preprocesar, no necesariamente como datos de entrenamiento.

## 3. Fuentes útiles y relación con requisitos

| Conclusión | Fuente | Tipo | Requisito al que apoya |
|---|---|---|---|
| La difusión ofrece "Calidad SOTA; control/edición" frente a "muestreo iterativo costoso"; los VAE son borrosos y las GAN sufren mode collapse | Session10.0 p7 | [MAT] | Justificar el modelo generativo (§3.3) |
| La temperatura y el muestreo equilibran "fidelidad al contexto y diversidad estocástica", análogo a strength/guidance/seed | Session10.0 p16–18 | [MAT] | Análisis fidelidad↔diversidad (§2.6) |
| Espacio latente/ELBO/VQ-VAE: base del VAE latente de Stable Diffusion | Session10.1 p5–29, p48–51 | [MAT] | Descripción del modelo (§4) |
| U-Net (denoiser de SD) | Session6.0 p19–20 | [MAT] | Descripción del modelo |
| CLIP/SigLIP: espacio compartido texto–imagen, "comparar directamente" | Session7.0 p58–61 | [MAT] | Métrica de cumplimiento del atributo (CLIPScore) |
| DINO: embeddings autosupervisados | Session9.0 | [MAT] | Métrica alternativa de diversidad |
| Bucle de agente OBSERVA→DECIDE→ACTÚA→REVISA | Sem10 Prompting p23 | [MAT] | Diseño del agente explicador con verificación (§3.5) |
| "Puede inventar una respuesta si carece de datos… puede necesitar supervisión" | Sem10 p25 | [MAT] | Verificador determinista del resumen |
| "La IA propone; las personas autorizan" | Sem10 p32 | [MAT] | Decisión humana final (§3.4) |
| **"Más agentes NO significa más inteligencia… usa la solución MÁS SIMPLE"; "si un script sencillo lo resuelve, evita construir 5 agentes"** | Sem10 p40–41 | [MAT] | Arquitectura de un solo agente LLM + verificador |
| Plantilla de prompt CONTEXTO / TAREA / RESTRICCIONES / FORMATO DE SALIDA; "añade… un mecanismo de verificación" | Sem10 p56, p60–61 | [MAT] | Prompt del explicador con salida JSON |
| Patrones secuencial, paralelo y supervisor; ReAct; LangGraph y otros frameworks | Sem10 p37–48 | [MAT] | Descartar frameworks pesados (no hacen falta) |
| Zero/one/few-shot | Sem9 p68 | [MAT] | Prompt del explicador (few-shot con ejemplo de JSON) |
| Calidad/sesgo de datos multimodales; "reproducir o amplificar sesgos" | Sem9 p40–44 | [MAT] | Ética y EDA del dataset (§2.4, §2.7) |
| Estereotipos y supervisión humana; SD como modelo abierto; caso Gemini (sobrecorrección) | Sem9 p71, p79, p81 | [MAT] | Reflexión ética (§2.7) |
| Salida JSON + validación de esquema (tasas "JSON válido" y "Schema válido") | `lora_eval_official.ipynb` celdas 0, 6 | [MAT] | Métrica de coherencia del resumen |
| Estructura de informe: restricción→decisión y métricas por dimensión técnica/decisión/social | `main (3).pdf` (informe previo del equipo, CS5363) | [MAT] | Informe de ≤5 págs. (§4) |
| LoRA reduce los parámetros entrenables | `2106.09685v2.pdf` | [MAT] | Solo si se llega al nivel avanzado (no recomendado) |
| FID, LPIPS, CLIPScore, ArcFace/FaceNet, ControlNet, InstructPix2Pix, LLM-as-judge | No están en el material | [EXT] | Métricas y alternativas; se citarán los papers originales |

## 4. Qué falta o queda sin verificar
- No hay material del curso sobre difusión aplicada, ControlNet ni métricas de modelos generativos: esas decisiones se apoyan en fuentes [EXT].
- No hay dataset en el proyecto. La imagen del notebook (`images.jpg`, un paisaje) es de origen y licencia desconocidos.
- Sobre el formato del video o la demo, el enunciado solo dice "presentación en vivo o un video".
