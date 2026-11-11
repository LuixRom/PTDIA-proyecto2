# Vitrina IA — Probador virtual de estilos

Proyecto 2 de *Planificación y Toma de Decisiones en IA* (CS5363, UTEC): sistema generativo interactivo con feedback
humano y explicación mediante agentes de IA.

El cliente de una tienda de moda en línea **sube su foto**, elige la línea (mujer u hombre) y una ocasión (trabajo,
reunión, evento o casual), y recibe **exactamente 5 looks** sobre sí mismo. **Acepta o rechaza** cada uno (por imagen o
por paquete), pide una **nueva ronda solo para los rechazados**, puede **restaurar** una ronda anterior del historial, y
un **agente LLM resume** sus decisiones. Un **verificador determinista** comprueba que el resumen coincide con el registro.

## Dos maneras de generar

| | Intermedio (local) | Básico (API) |
|---|---|---|
| Cómo | Segmentación de la ropa + *inpainting* de SD 1.5 + ControlNet de pose; solo se regenera la ropa | API de imágenes de OpenAI (`gpt-image-2`); se devuelven el rostro y el cabello originales |
| Costo | Gratis | De pago (≈ 0,023 USD por imagen, medido) |
| Semilla fija | Sí (modo prueba) | No |
| Foto | Se queda en tu equipo | Sale hacia OpenAI (casilla de consentimiento aparte) |
| Tiempo (5 looks) | 2 a 5 min con 4 GB de VRAM | ≈ 2 min |

El enfoque API solo aparece si hay una clave en `.env` (ver más abajo). Hay además un experimento de nivel avanzado
(un LoRA propio) que **no se adoptó**; ver «Nivel avanzado».

## Cómo ejecutarlo

### En tu equipo (recomendado para la demo)
```powershell
# una sola vez (en el .venv del proyecto)
.venv\Scripts\python.exe -m pip install "gradio==6.15.2"
.venv\Scripts\python.exe tools\download_models.py --llm      # modelos a la caché local (usa certificados de Windows)

# cada vez
.venv\Scripts\python.exe tools\app_local.py                  # abre http://127.0.0.1:7860
```
- Pensada para una GPU de 4 GB: Stable Diffusion con descarga a CPU y el LLM solo mientras redacta el resumen.
- **Cierra Chrome y otros programas pesados** antes de arrancar: carga varios modelos y, con poca RAM libre, Windows
  cierra el proceso sin mensaje (se ve el prompt de vuelta durante «Loading pipeline components»). Hacen falta unos 8 GB libres.
- Las sesiones se guardan fuera del repositorio, en `C:\PTDIA\sesiones_local` (contienen fotos de personas: bórralas al terminar).

### Enfoque API (opcional, de pago)
1. Copia `.env.example` a `.env` y pon tu clave (`OPENAI_API_KEY=...`; también se acepta el nombre `API-KEY`). **El `.env` está en
   `.gitignore`: nunca lo subas ni pegues la clave en un chat.**
2. Pon un límite de gasto en tu cuenta de OpenAI y revoca la clave al terminar las pruebas.
3. Al abrir la app debe imprimir «Enfoque API: disponible». En la pantalla elige «Básico · API de imágenes» y marca la casilla de consentimiento.
4. Prueba suelta sin la app: `.venv\Scripts\python.exe tools\probar_api_gpt.py --confirmo` (envía la foto y cuesta dinero).

### En Colab o Kaggle
Sube `vitrina_colab.ipynb` (lleva el código incluido), activa una GPU y ejecuta las celdas en orden. Detecta Kaggle solo.
No lleva la clave de la API, así que allí solo está el enfoque intermedio.

### Sin GPU (modelos simulados)
```powershell
python -m p2.vitrina_ui --mock
```

## Nivel avanzado: LoRA «Estilo de la tienda» (experimento)
Catálogo **sintético** generado con SD 1.5 (sin fotos de terceros) y un LoRA entrenado en Colab/Kaggle.
```powershell
.venv\Scripts\python.exe tools\gen_catalogo_lora.py --piloto   # genera imágenes en C:\PTDIA\lora_data (fuera del repo)
python tools\curar_lora.py piloto                              # aplica la curación manual (config\lora_curado_piloto.yaml) y arma el zip
# entrenar: subir el zip y abrir lora_entrenamiento_kaggle.ipynb en Kaggle o Colab (GPU T4)
.venv\Scripts\python.exe tools\eval_local.py FOTOS SALIDA --lora RUTA\pytorch_lora_weights.safetensors   # comparar con y sin LoRA
```
Resultado: se deforma el calzado a partir de los 400 pasos, hay fuga de identidad y, aplicado al probador, no mejora
el resultado; por eso la app no lo usa. El código y los resultados quedan como análisis (ver el informe).

## Estructura
```
p2/                paquete
  vitrina.py       motor: looks, máscaras, rondas, historial, métricas, exportación
  tryon.py         segmentación, pose, inpainting, accesorios, limpieza de fondo
  api_images.py    enfoque API (lee la clave de .env, sin imprimirla)
  vitrina_ui.py    interfaz web (Gradio)        vitrina_theme.py  identidad visual (cinta métrica) + fuentes en assets/
  guardrails.py    consentimiento, términos vetados, etiqueta     lora.py  carga/descarga de un LoRA sin peft
  explainer.py, verifier.py, llm.py, decisions.py, schemas.py ...   (agente, verificador y registro)
config/            vitrina.yaml (catálogo, ocasiones, modelos, guardrails), lora_*.yaml
tools/             app_local.py, download_models.py, eval_local.py y otras herramientas de evaluación; generadores de notebooks
tests/             pruebas automáticas
informe/           informe LaTeX (latex/main.tex), guion del video y guion de la demo
planificacion/     documentos de planificación y de la versión 0
vitrina_colab.ipynb · lora_entrenamiento_kaggle.ipynb · run_colab.ipynb (piloto de la versión 0)
```

## Pruebas y notebooks
```powershell
python -m pytest -q                      # pruebas locales (las de navegador usan Playwright y se omiten si falta)
python tools\build_vitrina_notebook.py   # regenerar vitrina_colab.ipynb (lleva el código incluido)
python tools\build_notebook.py           # regenerar run_colab.ipynb (versión 0)
python tools\build_lora_notebook.py      # regenerar lora_entrenamiento_kaggle.ipynb
```
Si cambias algo dentro de `p2/`, `tools/`, `config/` o `tests/`, **regenera los notebooks** y vuelve a correr las pruebas
(los tests fallan si un notebook quedó desactualizado). Dos pruebas pueden fallar solo en equipos con otra versión de
OpenCV o de torch fuera del `.venv`; no dependen del proyecto.

## Datos, privacidad y seguridad
- **No se suben fotos de personas reales de terceros.** Las fotos de prueba y las sesiones viven fuera del repositorio.
- La app no guarda tu foto: solo una copia reducida mientras dura la sesión, con un botón para borrarla.
- Cada imagen lleva la etiqueta «Simulación IA».
- La clave de la API vive solo en `.env` (ignorado por git); los mensajes de error enmascaran cualquier fragmento de clave.
- `Material/` y `doc-principal/` (material del curso) no se publican.

## Informe y guiones
El informe es `informe/latex/main.tex` (PDF compilado: `informe/informe.pdf`, 9 páginas; el límite es 10). Se compila con
pdfLaTeX (MiKTeX, Prism u Overleaf) **dos veces**, desde `informe/latex`, junto con la carpeta `figs/`:
```powershell
cd informe\latex
pdflatex -interaction=nonstopmode main.tex   # repetir una vez más para resolver las referencias
```
Las figuras se regeneran con `python informe/latex/make_figs.py`. Guiones en `informe/guion_video.md` (3 min) e
`informe/guion_demo.md` (6 min).

## Licencias y créditos
Stable Diffusion 1.5 (CreativeML OpenRAIL-M), ControlNet OpenPose (OpenRAIL), SegFormer B2 de ropa (licencia por verificar
en su ficha), ViTPose (Apache-2.0), CLIP (MIT en el código), Qwen2.5 (Apache-2.0), Phi-3.5-mini (MIT). Tipografías Big
Shoulders Display y Hanken Grotesk (SIL OFL 1.1) incluidas en `p2/assets/fonts`. Las licencias son las declaradas en las
fichas de cada modelo y no son una verificación legal.
