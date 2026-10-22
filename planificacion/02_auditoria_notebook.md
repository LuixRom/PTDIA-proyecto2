# 02 — Auditoría de `Untitled1.ipynb`

El notebook no se ejecutó ni se modificó. Lo analizado son el código y las **salidas guardadas** (texto e imágenes embebidas, extraídas a un directorio temporal e inspeccionadas visualmente). Los `execution_count` están vacíos, así que el **orden real de ejecución no es verificable**.

## 1. Objetivo aparente
Prototipar en Colab (T4) un generador de 5 variaciones con SD 1.5 img2img, comparar Diffusers frente a ComfyUI con SSIM y montar una evaluación humana ciega aceptar/rechazar. Además explora Qwen2-VL-2B como descriptor y juez de imágenes.

## 2. Entorno medido (salidas del notebook)
- GPU Tesla T4, 14.56 GB VRAM (celdas 1, 30); RAM 13.6 GB (celda 33); Python 3.13 (Colab).
- PyTorch 2.11.0+cu130 → 2.14.1 tras `pip install -U torch` (celda 3). Esa actualización rompe torchvision ("requires torch==2.11.0"). Diffusers 0.40.0, FastAPI 0.141.1.
- Checkpoint SD 1.5 `v1-5-pruned-emaonly.safetensors`: 3.97 GB (celda 32). Hay descargas sin `HF_TOKEN` (aviso de límite de velocidad).

## 3. Revisión celda a celda

| Celdas | Qué hace | Estado / salida real | Problemas |
|---|---|---|---|
| 0–1 | `pip install` diffusers, transformers, fastapi…; imprime versiones | OK | Sin versiones fijadas |
| 2 | Carga `stable-diffusion-v1-5/stable-diffusion-v1-5` con `AutoPipelineForImage2Image`, fp16, `enable_model_cpu_offload` | OK | `torch_dtype` deprecado (aviso); offload innecesario en T4 (más lento) |
| 3 | `pip install -U transformers torch accelerate` | Conflicto torch/torchvision | **Rompe la reproducibilidad**; eliminar |
| 4–5 | Comprobaciones | OK | — |
| 6 | Sube `images.jpg` (paisaje) y aplica `thumbnail(512,512)` → 512×341 | OK | Licencia y origen desconocidos; 341 no es múltiplo de 8 → salida recortada a 512×336 |
| 7 | 1 img2img, strength 0.40, 20 pasos, cfg 7, seed 12345 | 6.49 s | Prompt genérico |
| 8 | Guarda el original y la variación 1 | OK | — |
| 9 | 5 variaciones (seeds 12345–12349), CSV + JSON de configuración | 3.1–5.3 s/img | El código de la celda está truncado tras `json` (la escritura de `configuracion.json` no consta) |
| 10 | Barrido strength 0.20/0.40/0.65 × 5 seeds | 15 imágenes, 58.08 s en total | Ver el análisis visual |
| 11 | strength 0.55 × 5 seeds | ~3.4–3.9 s/img | — |
| 12, 28, 29 | Instala ComfyUI (**3 veces**, código idéntico) | OK | Duplicado; el comentario dice "No ejecutes esto en el notebook de Diffusers", pero se ejecutó en el mismo |
| 13–27 | Qwen2-VL-2B-Instruct: descripción few-shot, zero-shot, "objetiva" y autoevaluación | Salidas en español | El VLM califica **5/5 en todos los criterios** a ambas descripciones (celda 25) e infiere "otoño" o "luz débil" aunque se le pidió no hacerlo (celda 27) |
| 30–35 | Lanza el servidor ComfyUI, descarga el checkpoint, comprueba nodos | API disponible | La imagen ahora es la versión 678×452 (no la de 512) |
| 36 | ComfyUI img2img, euler/normal, denoise 0.40, 5 seeds | 672×448; 30.7 s la 1.ª, ~5.0 s el resto | — |
| 37–40 | SSIM vs original y 1−SSIM entre pares, ComfyUI y Diffusers (carpeta de la celda 9) | Ver la tabla de métricas | Se comparan imágenes de distinta resolución de origen (512 vs 678) reescaladas |
| 41 | Repite ComfyUI "controlado" | 5.19 s/img medio | Mismas imágenes que la celda 36, salvo la 1.ª (determinista) |
| 42–43 | Detiene ComfyUI | OK | — |
| 44 | Diffusers con **EulerDiscreteScheduler**, 672×448, mismos params y seeds | 6.04 s/img medio (con offload) | Comparable con ComfyUI salvo por el RNG (CPU frente a GPU): **las mismas semillas no dan el mismo ruido** |
| 45 | `evaluar_carpeta`: SSIM medio y diversidad para las dos carpetas "controladas" | Tabla comparativa | Reutilizable |
| 46 | UI ipywidgets ciega: 10 imágenes barajadas (seed 42), Radio aceptar/rechazar + motivo, botón Guardar | Se muestran 10/10 | El sistema queda oculto, bien. Los valores de los widgets no se guardaron |
| 47–51 | Diagnóstico: el CSV no existe; "Falta evaluar estas imágenes: [1…10]" | **Fallo** | Los widgets se re-renderizaron sin selección o se perdió el estado |
| 52 | Agrega `feedback_humano.csv`: ComfyUI 4/5, Diffusers 3/5 aceptadas | Tabla | **No trazable**: ninguna celda guardada muestra la creación de ese CSV (47–51 dicen que no existe; 53 se interrumpe) |
| 53 | Alternativa con `input()` | KeyboardInterrupt en la imagen 1 | — |
| 54–60 | Diffusers strength 0.60, 5 imágenes en `/content/diffusers_strength060` | Guardadas, **no evaluadas** | — |
| 61–66 | Relanzar ComfyUI | `ConnectionError` (celdas 63, 65); la 64 queda bloqueada en primer plano | Celdas muertas |

## 4. Resultados medidos (tal como figuran en las salidas)

| Configuración | Resolución | Tiempo/img | SSIM vs original | Diversidad (1−SSIM por pares) | Fuente |
|---|---|---|---|---|---|
| Diffusers, PNDM por defecto, strength 0.40 | 512×336 | 3.1–5.3 s | 0.4073 | 0.6599 | celdas 9, 40 |
| ComfyUI euler, denoise 0.40 | 672×448 | ~5.0 s | 0.4946 | 0.4759 | celdas 36, 38–39, 45 |
| Diffusers Euler, strength 0.40 | 672×448 | 6.04 s | 0.5280 | 0.4285 | celdas 44, 45 |
| Aceptación humana (no trazable) | — | — | — | — | celda 52: ComfyUI 80 %, Diffusers 60 % |

## 5. Análisis visual de las imágenes embebidas
Revisadas: el original (celda 6) y muestras de las celdas 9, 10 (strength 0.20 y 0.65), 36 y 44.

- **Original**: lago con reflejo de montañas, bosque otoñal y niebla. La estética sugiere una imagen ya generada o muy editada.
- **Diffusers PNDM a 512×336 (celdas 9–11)**: calidad **mala en todos los strengths**.
  - A 0.20 ya hay sobresaturación, texturas "quemadas" y ruido de alta frecuencia.
  - A 0.40, seed 12347, aparece un **pato/ave enorme** alucinado en el centro.
  - A 0.65 la escena se reinterpreta como un óleo con una barca: la composición se pierde.
  - Su "diversidad" SSIM alta (0.66) refleja artefactos, no variedad útil.
- **ComfyUI euler y Diffusers Euler a 672×448 (celdas 36, 44)**: fieles, limpias y coherentes, con fidelidad alta. Las 5 variaciones son **casi indistinguibles** a simple vista: solo cambian detalles de niebla y árboles. **No cumplen la intención de ofrecer 5 alternativas útiles y distintas.**
- [HIP] La degradación de la primera tanda se atribuye a la combinación del scheduler PNDM por defecto con pocos pasos efectivos (strength × 20) y a la entrada reducida a 512×341. No se ha aislado cuál de los dos factores pesa más. **Prueba pendiente:** mismo input, PNDM frente a Euler a resolución fija.

Conclusiones:
1. La métrica SSIM por sí sola **premia o castiga sin distinguir artefacto de variación semántica**. Hacen falta métricas perceptuales/semánticas (LPIPS, CLIP) y revisión humana.
2. La diversidad útil no se consigue solo con semillas: hay que **condicionar cada variación a un atributo distinto** (prompt, control).

## 6. Hardware
- Lo demostrado: T4 de 15 GB con SD 1.5 fp16 a ~5 s/img (672×448, 20 pasos, strength 0.4 → ~8 pasos efectivos).
- Local (medido en esta auditoría): RTX 3050 Laptop de **4 GB VRAM**, 15.7 GB de RAM y 22 GB libres en C:.
  - [EST] SD 1.5 fp16 con `enable_sequential_cpu_offload` o attention slicing debería caber, pero varias veces más lento. ControlNet añade ~0.7 GB fp16 y queda muy justo.
  - Decisión del usuario: **Colab T4 como entorno principal**.

## 7. Conservar / mejorar / descartar

| Elemento | Decisión | Justificación |
|---|---|---|
| Pipeline Diffusers img2img SD 1.5 fp16 | **Conservar** | Funciona, es reproducible (mismos bytes con la misma seed: celdas 7/9/10) y rápido en T4 |
| `EulerDiscreteScheduler` y resolución múltiplo de 8 | **Conservar** | Es la única configuración del notebook con calidad aceptable |
| Bucle seeds + registro de parámetros en CSV/JSON (9–11) | **Mejorar** | Pasar a JSONL con versión del modelo, scheduler, VRAM pico, prompt y negative prompt |
| `evaluar_carpeta` (45) | **Mejorar** | Añadir LPIPS, CLIPScore e identidad facial; igualar resoluciones antes de comparar |
| Evaluación ciega barajada (46) | **Conservar la idea** | Reimplementar en Gradio, con estado persistente |
| `enable_model_cpu_offload` en T4 | Mejorar | Innecesario con 15 GB; `pipe.to("cuda")` es más rápido |
| ComfyUI (12, 28–43, 61–66) | **Descartar** | Duplica Diffusers sin ventaja medida (SSIM y tiempos similares); servidor frágil en Colab (caídas en 63–65); difícil de integrar con la UI |
| `pip install -U torch` (3) | Descartar | Rompe torchvision |
| Thumbnail 512 sin alinear / PNDM | Descartar | Origen probable de los artefactos |
| Qwen2-VL-2B como juez (25) | **Descartar como juez** | Sin capacidad discriminante (5/5 a todo) y alucina atributos |
| ipywidgets/`input()` como UI | Descartar | Perdió estado; no apto para la demo |
| CSV de la celda 52 | No usar como evidencia | Origen no trazable |

## 8. Pruebas que faltan
1. PNDM frente a Euler a resolución fija (aislar la causa de los artefactos).
2. Cualquier prueba con **rostros** (el dominio elegido): identidad, deformación facial, safety checker.
3. Variaciones condicionadas por atributo (5 prompts distintos) y ControlNet.
4. Métricas LPIPS, CLIPScore e identidad facial.
5. Evaluación humana real y trazable.
6. Agente LLM de resumen y su verificación (no existe en el notebook).
7. Medición de VRAM pico (`torch.cuda.max_memory_allocated`).
