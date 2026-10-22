# Proyecto 2 — Sistema generativo interactivo con feedback humano

Curso Planificación y Toma de Decisiones en IA (UTEC).

Dada una imagen de entrada, el sistema genera **exactamente 5 variaciones**. El usuario **acepta o rechaza** cada una, las decisiones se **registran** de forma estructurada, y un agente LLM las **resume**. Un **verificador determinista** independiente comprueba que el resumen coincide con el registro.

> Estado: **preparación del piloto**. El código está listo y probado localmente con datos simulados.
> Todavía no se ha descargado ningún modelo ni imagen y no se ha ejecutado ninguna generación.
> Planificación y licencias: `planificacion/` (en especial `10_verificaciones_y_plan_piloto.md`).

## Para el equipo (flujo de trabajo)
```powershell
git clone <url-del-repo>; cd proyecto-2
python -m pytest -m "not gpu" -q        # pruebas locales (sin GPU ni modelos)
```
- Si cambias código, configuración o datos, **regenera el notebook**, que lleva el código incluido: `python tools/build_notebook.py`. Después vuelve a ejecutar los tests y sube **también** `run_colab.ipynb`. El test `test_notebook_is_up_to_date` falla si se olvida.
- Para probar en Colab basta con subir `run_colab.ipynb` (sección «Google Colab»).
- Para probar el notebook entero en local sin descargar modelos: `.venv\Scripts\python.exe tools\simulate_colab.py`. Requiere un entorno con torch, diffusers y transformers.
- No subas fotos de personas reales a `data/`: `tools/make_zip.py` y `tools/build_notebook.py` rechazan imágenes no registradas.
- Resultados y análisis de los pilotos y de la app: carpeta `results/` (`piloto1_analisis.md`, `piloto2_analisis.md`, `app_sesiones_analisis.md`).
- No incluidos en el repositorio: `Material/` y `doc-principal/` (material del curso).

## Estructura
```
p2/            paquete: config, env_check, preprocess, generate, metrics, schemas,
               decisions, verifier, explainer, ui, export
experiments/   pilot.py (A1 vs A2), blind_eval.py (evaluación humana ciega)
tools/         select_images.py, build_notebook.py, make_zip.py
config/        pilot.yaml  ← modelos (con revisión fijada), parámetros, semillas, prompts
requirements/  colab.txt (lo único instalable), colab-expected.txt, local-tests.txt
data/sfhq_t2i/ imágenes de entrada (vacío hasta su aprobación) + licencia + manifiesto
tests/         pruebas locales (sin GPU ni modelos)
run_colab.ipynb  notebook del piloto (generado por tools/build_notebook.py)
```

## Windows: desarrollo y pruebas (sin GPU, sin descargas)
Requisitos: Python ≥ 3.10 con `pydantic`, `pytest`, `pyyaml`, `pillow`, `numpy`. Opcionales: `scikit-image` y `opencv-python` (métricas y detección de caras), `gradio` (UI). En el equipo del autor ya estaban instalados (ver `requirements/local-tests.txt`).

```powershell
python -m pytest -m "not gpu" -q          # pruebas locales
python -m p2.ui --mock                    # UI de aceptar/rechazar con generador simulado
python tools/build_notebook.py            # regenerar run_colab.ipynb tras cambiar sus celdas
python tools/make_zip.py                  # crear dist/proyecto2_colab.zip
```

Qué cubren las pruebas:
- Esquemas (exactamente 5 variaciones, IDs válidos).
- Registro de decisiones (aceptar, rechazar, cambio de opinión, escritura atómica, no sobrescritura).
- Verificador del resumen (ID intercambiado, omitido o inventado; estado contrario; casos 0/5 y 5/5).
- Explicador con LLM simulado (reintentos, respaldo por plantilla, motivo con instrucciones inyectadas).
- Preprocesamiento (formatos, EXIF, tamaños).
- Generador simulado.
- Piloto completo con backend `mock` (registro, reanudación, métricas, conjunto ciego, manifiesto).
- Comprobación de recursos y de compatibilidad de dependencias.
- Selección de imágenes (con y sin CSV), registro de la imagen de entrada y rechazo de imágenes no registradas en el ZIP.
- Invariantes del notebook, de la configuración y del ZIP generado.

## Imagen de entrada (1 imagen sintética SDXL de SFHQ-T2I)
En PowerShell, desde `C:\PTDIA\proyecto-2`:

1. **Candidatas.** Solo lista nombres de archivo; no descarga imágenes:
   ```powershell
   python tools/select_images.py --kaggle-list --n 10
   ```
   Si tiene el CSV oficial, añada `--csv C:\PTDIA\sfhq_meta\SFHQ_T2I_dataset.csv` para filtrar además por estilo fotográfico. Guárdelo **fuera** de `data/`.
2. **Elegir y descargar solo esa imagen.** Recorra las candidatas en orden y quédese con la primera que, a la vista, tenga 1 cara, sea fotorrealista, esté de frente o en 3/4 y no tenga oclusión fuerte. En [Kaggle](https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models) › Data › `a_small_sample_new/`, descargue ese archivo a `data\sfhq_t2i\` sin renombrarlo.
3. **Registrarla.** La herramienta valida el nombre, JPEG 1024×1024, el modelo permitido, el manifiesto y, con `--csv`, el motor y el estilo. Después calcula el SHA-256, añade la fila a `MANIFEST.csv` y fija `dataset.pilot_image`:
   ```powershell
   python tools/register_image.py data\sfhq_t2i\SDXL_image_NNNNNNN.jpg --pilot --motivo "1 cara, frontal, fotorrealista, sin oclusión"
   ```
4. **Notebook y ZIP.** Se rechazan si en `data/sfhq_t2i/` hay algo sin registrar:
   ```powershell
   python tools/build_notebook.py
   python -m pytest -m "not gpu" -q
   python tools/make_zip.py
   Get-FileHash dist\proyecto2_colab.zip -Algorithm SHA256
   ```
5. Suba `run_colab.ipynb` a Colab (sección siguiente): ya incluye la imagen y el código.

Licencia: se registra la que declara la ficha de Kaggle (MIT; aviso en `LICENSE_MIT.txt`). **No es una verificación legal.** Detalle en `planificacion/10` §1. Sin CSV, el modelo se toma del prefijo del nombre y el estilo queda como «no verificado». Alternativa: SFHQ parte 3 (CC0).

## Google Colab: piloto A1 vs A2
**Solo hay que subir `run_colab.ipynb`**: el código y la imagen de entrada van incluidos en él (celda 1b, con SHA-256). No hay que subir ningún ZIP. `dist/proyecto2_colab.zip` queda como alternativa (`FUENTE_CODIGO = "subir"` o `"drive"`).

1. https://colab.research.google.com → **Archivo › Subir cuaderno** → `run_colab.ipynb`.
2. **Entorno de ejecución › Cambiar tipo de entorno de ejecución › GPU T4**.
3. **Entorno de ejecución › Ejecutar todas.** Sin autorizaciones (por defecto):
   - Las celdas 1–5 comprueban la GPU, la RAM y el disco, validan la compatibilidad de los paquetes (sin red y sin pip) y verifican la imagen de entrada.
   - Las celdas 6–8 muestran «⏸ omitida» y la ejecución termina sin errores.
4. Para ejecutar el piloto, marque en la celda 1 `AUTORIZO_DESCARGAR_MODELOS` y `AUTORIZO_EJECUTAR_PILOTO` y vuelva a **Ejecutar todas**.
   - Se descargan SD 1.5 fp16 y CLIP con la revisión fijada (≈3.4 GB, estimado).
   - Se hacen 12 generaciones: 1 de calentamiento, A1 ×5, A2 ×5 y 1 repetición de determinismo.
   - Se calculan las métricas (sin LPIPS, que no está instalado en Colab: `ACEPTO_SIN_LPIPS = True`).
   - Se descarga un ZIP de resultados.

Garantías del notebook:
- No instala ni cambia paquetes (PyTorch, torchvision, transformers y huggingface-hub nunca se tocan).
- Cada celda comprueba que se ejecutaron las anteriores y, si no, lo explica en vez de dar un error críptico.
- Las celdas 6 y 7 exigen la validación de la celda 4, para la misma configuración y sin cambios de paquetes.

### App final (celdas 9 y 10)
1. En la celda 1, marque `AUTORIZO_DESCARGAR_MODELOS` y `AUTORIZO_APP` (el piloto puede quedar desmarcado) y use **Ejecutar todas**.
2. La celda 9 carga SD 1.5 y el LLM **Qwen2.5-1.5B-Instruct** (Apache-2.0; ≈3.1 GB la primera vez) y muestra la app **dentro de la celda**, sin enlace público:
   - confirmar el consentimiento y subir una imagen (por ejemplo `data/sfhq_t2i/SDXL_image_0035846.jpg`);
   - generar 5 variaciones (A2: luz cálida, luz fría, acuarela, sonrisa, B/N; strength 0.55, elegido en los pilotos);
   - aceptar o rechazar cada una, con un motivo opcional;
   - pulsar «Generar resumen»: lo redacta el LLM y lo comprueba el verificador determinista. Si no coincide con las decisiones, se usa la plantilla y queda marcado.
3. La celda 10 descarga `sesiones_app_….zip` con `decisions.json`, `events.jsonl` y `summary.json` de cada sesión. Esos datos sirven para medir la coherencia del resumen y para el informe.

### Antes de subirlo: simulación completa en Windows
`tools/simulate_colab.py` ejecuta las **celdas reales** del notebook con el `.venv` (mismas versiones que Colab: torch 2.11, diffusers 0.41.0, transformers 5.20.2, huggingface-hub 1.33.0).
- Usa un sustituto de `google.colab` y modelos **diminutos** creados en local con las mismas clases (no descarga nada).
- Comprueba cuatro escenarios: por defecto, piloto completo, celdas fuera de orden y ZIP subido.

```powershell
python tools/build_notebook.py
.venv\Scripts\python.exe tools\simulate_colab.py      # debe terminar con "TODO COMO SE ESPERABA"
python -m pytest -m "not gpu" -q
python tools/make_zip.py
```

## Métricas e interpretación
- **SSIM**: solo referencia. En `Untitled1.ipynb` premió artefactos como si fueran "diversidad".
- **LPIPS**: distancia perceptual.
- **Similitud CLIP**: semántica, **no identidad**.
- **CLIPΔ**: si se aplicó el atributo pedido.
- **Diversidad**: media de los 10 pares.
- **Fallos**: caras detectadas (OpenCV Haar), NSFW, imagen negra.
- **Identidad y calidad**: se juzgan en la **evaluación humana ciega** (`experiments/blind_eval.py`).
- Con 1 imagen de entrada los resultados son **exploratorios**: no son evidencia estadística generalizable.

## Ética y licencias (resumen)
- SD 1.5: CreativeML Open RAIL-M. Prohibido usarlo para discriminar, dañar menores, difundir datos personales o desinformación.
- El safety checker permanece activo.
- CLIP se usa solo en la evaluación offline y nunca para identificar personas.
- Las imágenes de entrada son sintéticas. Las variaciones se etiquetan como generadas por IA.
- La UI exige consentimiento antes de subir una imagen.
- Sin APIs de pago ni credenciales en el código.
