# Sesiones de la app en Colab — medición de referencia (antes de mejorar el explicador)

Fuentes (manifiestos íntegros):
- `results/sesiones_app_20261022-0018/` (1 sesión; ZIP SHA-256 `93b4a6af…eb84`)
- `results/sesiones_app_20261022-0122/` (4 sesiones; ZIP SHA-256 `50b0cd8d…a5d1`)

Configuración: A2 (5 atributos), strength 0.55, LLM Qwen2.5-1.5B-Instruct (greedy). Estas 5 sesiones se hicieron con el
notebook **anterior** a la mejora del explicador (sin `raw_responses` en summary.json; prompt sin formato fijo ni bucle de
revisión). Sirven como línea base.

## Resultados (n = 4 sesiones completas, 20 variaciones; exploratorio)
| Métrica | Valor |
|---|---|
| Sesiones completas / abandonadas | 4 / 1 (`s-a8ed0d72`: 2 de 5 imágenes bloqueadas por el filtro de seguridad) |
| Aceptación global | 14/20 = 70 % |
| Aceptación por atributo | luz cálida 3/4 · luz fría 3/4 · acuarela 3/4 · **sonrisa 2/4** · B/N 3/4 |
| Resumen escrito y aprobado por el LLM | **1/4 = 25 %** (resto: plantilla de respaldo) |
| Coherencia final mostrada al usuario | 4/4 (el verificador nunca dejó pasar un resumen incoherente) |
| Imágenes marcadas NSFW (falsos positivos con caras) | 2 de 25 generadas (8 %) |

## Fallos del LLM detectados por el verificador
- Sesión 0018: "V1, V2, V3: el resumen no indica explícitamente su estado" (no usó "aceptada").
- `s-98267531`: **decisión equivocada**: dijo que V1 fue rechazada cuando fue aceptada.
- `s-20e0f31c` (5/5 aceptadas): no mencionó ningún identificador V1..V5.
- `s-3df0866c`: aprobado al 1.er intento, pero con errores de contenido que el verificador no evalúa
  ("luz fría estudiada", "época bética"): el verificador comprueba identificadores y estados, no la calidad del texto.

## Lectura
- El diseño "LLM redacta, verificador decide" evitó mostrar resúmenes falsos (p.ej. V1 rechazada) → explicabilidad segura,
  pero con Qwen2.5-1.5B sin formato fijo la tasa de resúmenes propios del LLM fue baja (25 %).
- Con greedy, los 3 intentos eran idénticos: el reintento no aportaba nada.
- Mejoras ya implementadas (pendientes de medir con nuevas sesiones): formato fijo + ejemplo (few-shot), reintento con los
  errores del verificador, registro de las respuestas del LLM, aviso visible cuando el filtro de seguridad bloquea una imagen.
- Ética/datos: varias imágenes de prueba eran fotos de internet de personas reales (bancos de imágenes). Se movieron fuera
  del proyecto (`C:\PTDIA\imagenes_prueba_app\`) y no se incrustan en el notebook. Para el informe, usar solo caras sintéticas
  o con consentimiento.

## Después de mejorar el explicador (notebook con formato fijo + ejemplo + bucle de revisión)
Fuente: `results/sesiones_app_20261022-0233/` (ZIP SHA-256 `2d7dc9e4…6262`, manifiesto íntegro; `raw_responses` presente
→ versión nueva confirmada).
- Sesión `s-583225be` (imagen piloto SDXL): V1–V3 aceptadas, V4–V5 rechazadas (cambio de género; sonrisa y ropa raras).
- Resumen **escrito y aprobado por Qwen2.5-1.5B al 1.er intento** (sin plantilla); re-verificado: coherente.
- Fidelidad de los motivos (revisión manual): atribuye "mejorar la ropa" a V1–V3 (el usuario no lo dijo) y omite
  "sonrisa rara" en V4 → el verificador garantiza decisiones correctas, **no** la exactitud de los motivos (limitación).
- n = 1: insuficiente para concluir que la mejora funciona (antes 1/4). Pendiente: sesiones con 5/5 aceptadas,
  mayoría rechazadas y otras caras.

## Sesiones de prueba del explicador (notebook nuevo) — `results/sesiones_app_20261022-0312/` (ZIP `401f1f10…9b32`, íntegro)
- `s-583225be` (imagen piloto) **reutilizada**: el usuario cambió las decisiones de la sesión anterior a 5/5 aceptadas y
  volvió a pedir el resumen (24 eventos en `events.jsonl` registran cada cambio con su hora; el resumen anterior se conserva
  en `sesiones_app_20261022-0233`). Prueba deliberada: decisiones no representativas de la opinión real.
- `s-8a7719e1` (imagen piloto): 0/5 aceptadas, prueba deliberada.
- `s-4f706022` (otra cara, foto de internet): decisiones reales, V1, V2, V4 aceptadas; V3, V5 rechazadas
  (V5: "parece otra persona"). Sus PNG no se publican en el repositorio (foto de un tercero).

| Sesión | Caso | Resumen final | Intentos | Observación |
|---|---|---|---|---|
| s-583225be (0233) | mixto 3/5 | **LLM** | 1 | motivos parcialmente infieles ("ropa") |
| s-583225be (0312) | 5/5 aceptadas | **LLM** | 2 | el intento 1 no nombró los IDs; **el bucle de revisión lo corrigió**; dice "todos los usuarios" (hubo 1) |
| s-8a7719e1 | 0/5 aceptadas | plantilla | 3 | el LLM **inventó aceptaciones** (V1–V4 o V4–V5) en los 3 intentos; el verificador lo bloqueó |
| s-4f706022 | mixto 3/5 | **LLM** | 1 | correcto; agrupa motivos (no menciona "dientes" en V4) |

**Comparación (exploratoria):** resumen propio del LLM aprobado **antes 1/4 (25 %) → después 3/4 (75 %)**; al primer
intento 2/4; el bucle de revisión rescató 1 caso. Resúmenes incoherentes mostrados al usuario: **0 en ambos casos**.
Caso más difícil para Qwen2.5-1.5B: **todas rechazadas** (sesgo a declarar aceptaciones). Limitaciones que el verificador no
cubre: fidelidad de los motivos y errores de redacción ("todos los usuarios").
Aceptación real con caras distintas de la piloto (sin sesiones de prueba): s-98267531 3/5, s-3df0866c 3/5, s-20e0f31c 5/5,
s-4f706022 3/5.
