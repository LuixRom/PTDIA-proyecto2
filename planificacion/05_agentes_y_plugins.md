# 05 — Plugins, skills, agentes y evaluador visual

Todo lo de este documento se comprobó en `~/.claude/settings.json`, `~/.claude/plugins/installed_plugins.json`, los manifiestos de cada plugin y la lista de skills y agentes de la sesión del 2026-10-21. No se instaló, desactivó ni modificó nada.

## 1. Inventario real de la sesión

| Plugin | Qué es realmente (evidencia) | Aporta al proyecto | Recomendación |
|---|---|---|---|
| `agent-skills` (addy, 0.6.13) | Skills de flujo de ingeniería: spec, plan, TDD, review, documentación/ADR, y agentes `code-reviewer`, `test-engineer` y `security-auditor` | Sí, en la implementación: TDD del verificador y del esquema, review final | Usar `test-driven-development` y `code-review-and-quality` puntualmente |
| `data-scientist` (magnus) | Skill de estadística y diseño experimental | Sí: análisis de acuerdo (κ, W), diseño de 04 | Cargar una vez al analizar los resultados del experimento |
| `ml-engineering` (magnus) | Skill de evaluación, eval-sets, reproducibilidad y despliegue de modelos | Parcial: diseño del eval-set y lineage | Opcional; **se solapa** con data-scientist en evaluación |
| `data-cleaning` (magnus) | Limpieza de datos tabulares/JSON | Casi nula (aquí los datos son imágenes) | No usar |
| `claude-code-setup` | Recomendador de automatizaciones de Claude Code | No | No usar |
| `figma` (MCP oficial) | Lectura y escritura de diseños Figma | Opcional para maquetar la UI; Gradio no lo necesita | No usar salvo que se pida un mockup |
| `filetree` | Mod de panel IDE (árbol de archivos) | Solo visual; no da información que no den `ls` o Glob | No necesario |
| `habblaud` | Mod que registra el uso del plan (5h/semanal) y consulta un servidor local `http://127.0.0.1:<port>/api/mod/summary` | **No.** No ahorra tokens: es telemetría y avisos de uso | Indiferente; no depender de él |
| `pets` | Mascota cosmética | No | — |
| `ponytail` (5.3.2) | Inyecta instrucciones de "cambio mínimo" en cada sesión y subagente. Su README declara −45 % de tokens y −53 % de código según **su propio benchmark** | Estilo de código conciso. El ahorro **no está verificado aquí** y la inyección tiene un coste de contexto fijo | Mantener si gusta el estilo; no presuponer ahorro |
| Graphify | Ejecutable `~/.local/bin/graphify.exe`, enganchado como **hook global** `hook-guard` en cada Read/Bash/Grep/Glob. **No existe `graphify-out/` en este proyecto** | No configurado aquí. Con 24 archivos y sin código propio no aporta frente a la lectura directa | No construir grafo ahora; reconsiderar cuando haya código (`app/`) |
| impeccable (skill de usuario + hooks) | Hooks en SessionStart/PostToolUse/Stop para revisión de UI | Podría revisar la UI web. Gradio genera su propio HTML | Opcional en la etapa de UI |

Solapes detectados:
- **Evaluación y experimentación**: `data-scientist` ≈ `ml-engineering` (eval-set) ≈ `agent-skills:spec/plan`. Para el diseño experimental basta `data-scientist`.
- **Planificación**: `agent-skills:planning-and-task-breakdown` ≈ el agente nativo `Plan`. Ninguno es necesario: el plan ya está en `07`.
- **Review de código**: `agent-skills:code-reviewer` ≈ `ponytail-review` ≈ la skill integrada `code-review`. Basta una.

**Ningún plugin puede ver ni evaluar imágenes.** La capacidad de visión la tiene el propio modelo de Claude Code (lectura de PNG/JPG), como se hizo en esta auditoría con las imágenes del notebook.

## 2. Agentes propuestos (mínimo necesario)
Principio del curso: "usa la solución MÁS SIMPLE que funcione… si un script sencillo lo resuelve, evita construir 5 agentes" (Sem10 p40–41).

| Agente | ¿Necesario? | Entradas | Entregable | Evitar duplicidad |
|---|---|---|---|---|
| Auditoría documental (Explore nativo) | **Ya usado** en la fase 1 (lectura de 19 PDFs, ~87 k tokens fuera del contexto principal) | PDFs | Resumen por archivo con páginas | No repetir: el resultado está en 01 |
| Revisión de notebook/entorno | No: se hizo inline | — | 02 | — |
| Investigador de modelos | No: la comparativa 03 se basa en la auditoría y la documentación | — | — | — |
| **Juez visual auxiliar** | **Sí, en el experimento** | Original + sets anonimizados (IDs aleatorios) + rúbrica fija | `judge_scores.jsonl` | Se ejecuta con IDs ciegos y no ve `blind_key.json` |
| Revisión de requisitos/reproducibilidad | Sí, una vez al final (agente `code-reviewer` o skill `code-review`) | Repo + 08 | Lista de huecos | Una sola pasada antes de entregar |

Coste de contexto [E]:
- Cada subagente arranca en frío (~10–30 k tokens de lectura).
- El juez visual con 20–30 imágenes supone ~1.5 k tokens por imagen en lectura más la rúbrica: unos 50 k por campaña de piloto.

## 3. Evaluador visual auxiliar: diseño

### Rúbrica fija (igual para el juez y para los humanos)
Por imagen, en escala 1–5 con justificación visual concreta obligatoria (qué zona y qué defecto):
1. **Identidad**: ¿es reconociblemente la misma persona?
2. **Calidad visual**: nitidez y anatomía (ojos, dientes, manos, simetría).
3. **Cumplimiento del atributo** (el que se pidió).
4. **Artefactos o cambios no deseados**: lista (fondo alterado, ropa, edad, tono de piel, objetos alucinados).

Por set de 5: diversidad (1–5), ranking de las 5, y marca de **empate o caso ambiguo** cuando la diferencia sea ≤ 0.5.

Salida JSON validada con un esquema:
```json
{"set_id":"S7","image_id":"k3f9","identity":4,"quality":3,"attribute":5,
 "artifacts":["dientes fusionados"],"evidence":"boca: incisivos unidos; fondo igual",
 "recommend":"aceptar","ambiguous":false}
```

### Opción A — modelo multimodal (VLM)
- **A-local, Qwen2-VL-2B**: evidencia en contra (notebook celda 25: 5/5 en todo, sin discriminar). Descartado.
- **A-local, VLM de 7B** (Qwen2-VL-7B / Qwen2.5-VL-7B en 4-bit, ~6 GB en T4) [E]: más capaz, pero no está validado aquí y añade descargas (~5 GB).
- **A-Claude (esta sesión)**: puede leer las imágenes (demostrado en esta auditoría) sin coste adicional de API dentro del plan del usuario. Es independiente del generador (SD 1.5).
  - Sesgos documentables: la preferencia estética de un modelo entrenado con RLHF, la sensibilidad a la estética "fotográfica", la evaluación de rostros de grupos poco representados y el sesgo de posición (mitigado barajando).
  - No forma parte del sistema entregable. Es una herramienta de análisis del experimento.

### Opción B — métricas + reglas + humanos
Las métricas de 04 §6, con reglas explícitas: identidad < umbral → "rechazo sugerido"; cara no detectada → fallo. Es explicable, reproducible y barata, pero ciega a defectos semánticos (dientes, ojos).

### Recomendación
**B como juez principal y automático (integrado en el sistema), con evaluación humana ciega como verdad de referencia. A-Claude como juez cualitativo auxiliar del experimento**, que aporta las observaciones visuales concretas que B no ve.
- Se mide el acuerdo de A y de B con los humanos (κ y correlación de Spearman). Ese acuerdo es en sí un resultado para la sección "Evaluación" de la rúbrica.
- La recomendación del juez (el mejor candidato) y la decisión final del usuario se guardan en **campos y archivos separados** (`judge_scores.jsonl` frente a `decisions.jsonl`).

### Interfaz ciega
Gradio con IDs aleatorios, orden barajado por una semilla registrada y sin metadatos visibles. El mapeo ID→método queda en un archivo que ni el juez ni los evaluadores abren hasta el final.


---

## Revisión 2 (2026-10-21)
Texto original conservado arriba. Detalle y fuentes en `10_verificaciones_y_plan_piloto.md`.
- CLIP queda restringido a la **evaluación offline** (ficha de OpenAI: "Any deployed use case… out of scope"; reconocimiento facial siempre fuera de alcance). No forma parte de la app final.
- El agente explicador y el verificador están en módulos separados (`p2/explainer.py` y `p2/verifier.py`). El verificador se probó primero con decisiones y resúmenes simulados: 88 tests locales en verde a 2026-10-21.
