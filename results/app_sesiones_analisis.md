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
