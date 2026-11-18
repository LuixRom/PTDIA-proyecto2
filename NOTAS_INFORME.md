# Notas para actualizar el informe (rama `estilo-usuario`)

## Qué cambió
- **El sistema aprende el estilo del cliente.** Después de decidir, «Deducir mi estilo» muestra dos listas editables,
  **Quiero** y **No quiero** (por ejemplo «corte recto», «color negro», «estilo formal»), y una explicación en español.
  «Nueva ronda con mi estilo» regenera los rechazados (o las 5) usando ese perfil. Cada look nuevo dice por qué se
  eligió («Elegido porque prefieres corte recto»).
- **Catálogo con etiquetas y más variedad.** Cada prenda tiene color, corte (recto, baggy, pitillo, entallado,
  holgado, regular) y estilo (formal, office casual, casual, elegante). Se agregaron pantalones recto, baggy y pitillo
  en las dos líneas y dos prendas de arriba. Ahora hay 43 prendas (14 de mujer y 29 de hombre) y 8 accesorios; el
  informe dice 33. Los 5 looks iniciales de cada ocasión mezclan cortes y colores a propósito: la ronda 1 explora.
  «Trabajo» se muestra como «Trabajo (office casual)».
- **Registro de pruebas.** El campo del alias ahora es «Nombre del participante (para el registro de pruebas)». La
  casilla de consentimiento se mantiene. `tools/exportar_pruebas.py` arma el Excel y un JSON con lo mismo.
- **Arreglos:**
  - Las nuevas rondas con la API fallaban: el contexto de la foto se reconstruía incompleto. En local se omitían los
    accesorios y la limpieza del fondo en la ronda 2.
  - Con fotos no verticales, la API recibía un tamaño de salida vertical y la imagen se deformaba.

## Cómo funciona el aprendizaje de estilo (para la sección de diseño e implementación)
1. **Perfil determinista.** Cada etiqueta de un look suma +1 si el cliente lo aceptó y −1 si lo rechazó, en todas
   las rondas. Si el total es positivo va a «Quiero»; si es negativo, a «No quiero». Cada preferencia cita sus
   evidencias (por ejemplo, V3-r1 es el look V3 de la ronda 1).
2. **El LLM lee los motivos y redacta.** Recibe los datos delimitados y devuelve JSON con preferencias y una
   explicación. Ejemplo de la profesora: «no me gustan los pantalones pitillo, me gustan los rectos o baggy» da
   *No quiero* corte pitillo y *Quiero* corte recto y corte baggy.
3. **Verificador** (mismo principio de siempre: el LLM redacta, no decide qué es verdad). Una preferencia del LLM
   solo se acepta si la respalda:
   - la etiqueta de un look con la decisión correspondiente (quiero = aceptado, no quiero = rechazado), o
   - el texto del motivo de ese look, sin contradecirlo («no me gustan los pitillos» no puede volverse «quiero pitillo»).

   También rechaza valores fuera de la lista y looks inventados. Si falla, se reintenta hasta 2 veces con los
   errores. Si sigue fallando, lo no respaldado se descarta y se muestra el perfil determinista, marcado como tal.
   Lo que el cliente dijo en un motivo pesa más que lo deducido de las etiquetas.
4. **El cliente corrige** las listas: la decisión final es suya.
5. **Selección de la ronda:**
   - candidatos: todos los looks de su línea, primero los de la ocasión;
   - se descarta cualquiera con algo de «No quiero» y la ropa ya mostrada;
   - se ordena por coincidencias con «Quiero»;
   - si no quedan candidatos, se ajusta el mejor look con el corte o color preferido.

   Todo queda en `history.jsonl` y `events.jsonl`: el perfil deducido, el editado y por qué se eligió cada look.

## Métricas nuevas (pestaña «Métricas de la sesión» y Excel)
| Métrica | Qué responde de la rúbrica |
|---|---|
| Aceptación por ronda (r1, r2, …): aceptados / decididos entre los looks **generados** en esa ronda | ¿La iteración mejora con el feedback? |
| % de looks de rondas con estilo que **respetan «No quiero»** | Coherencia entre preferencias e iteraciones |
| % de looks de rondas con estilo que **coinciden con algún «Quiero»** | Coherencia entre preferencias e iteraciones |
| % de looks regenerados que el cliente había rechazado | Correspondencia entre decisiones e iteraciones (100 % salvo que pida «Las 5») |
| Perfiles con `fuente = llm` frente a `perfil determinista` (hoja perfiles) | Qué tan seguido el LLM supera el verificador |

## Qué cifras sacar del Excel
Se genera con `.venv\Scripts\python.exe tools\exportar_pruebas.py C:\PTDIA\sesiones_local`.
- **Hoja `resumen`:**
  - número de participantes (mínimo 10);
  - **aceptación media de la ronda 1 frente a la ronda 2**: es la cifra principal de «aprendió mi estilo». Usen
    también la de la ronda 1 solo de las sesiones que tuvieron ronda 2, para comparar lo mismo;
  - % que respeta «No quiero» y % con algún «Quiero»;
  - SUS medio.
- **Hoja `perfiles`:** cuántas deducciones salieron con `fuente = llm` y cuántas con perfil determinista; una o dos
  explicaciones como ejemplo.
- **Hoja `decisiones`:** dos o tres motivos reales y cómo se tradujeron en Quiero / No quiero (ejemplo cualitativo).
- Reportar todo como **exploratorio** (n ≈ 10, sin inferencia estadística).

## Decisiones tomadas ante ambigüedades (mencionar en limitaciones)
- Las preferencias usan un vocabulario fijo (las etiquetas del catálogo). Lo que el cliente diga fuera de ese
  vocabulario («brillos», «pastel») no se usa para elegir.
- Por defecto se regeneran solo los rechazados; «Las 5» es opcional.
- El verificador comprueba cada preferencia estructurada. En el texto libre de la explicación solo comprueba que no
  cite looks inexistentes; la polaridad de cada frase no se verifica.
- El perfil por etiquetas también suma lo que no motivó la decisión: si rechazó un look por el pantalón, la blusa
  también resta. Por eso existen los motivos y la edición manual.

## Limitaciones a declarar
- Las 8 prendas nuevas **no se validaron todavía con los modelos reales**.
- En el enfoque intermedio, un pantalón baggy no puede salir del contorno del pantalón original, porque la máscara
  cubre solo la ropa detectada. Con la API sí se ve el corte. Para la demo conviene la API.
- El Excel y las sesiones contienen **nombres y fotos de participantes**: no subirlos al repositorio. Pedir
  consentimiento a cada persona y borrarlos al terminar.

## Estado de las pruebas automáticas
- 257 pasan y 3 se omiten (playwright no instalado).
- 1 falla desde antes de estos cambios: `test_halo_of_old_silhouette…`, porque OpenCV no está en el `.venv`.
- `tests/test_lora.py` necesita torch.

## Validación de las prendas nuevas con los modelos reales (rama `estilo-usuario-mejoras`)
Probado en una RTX 2060 (6 GB) con `tools/eval_local.py`, foto `p2/assets/ejemplos/pose_ejemplo.jpg` (cuerpo entero, jean
celeste), ocasiones «trabajo» y «casual» de la línea mujer, modo prueba (semillas fijas). Falta la línea hombre: no hay
una foto de cuerpo entero de hombre en el repositorio.

**Problema encontrado.** En los looks de dos prendas (arriba + pantalón) el color salía mal casi siempre, sobre todo
hacia el azul: pantalón negro → azul petróleo, pantalón ancho beige → celeste, pitillo gris → azul marino, polo negro →
blanco. Los looks de una pieza (traje beige) salían bien.

**Causa, comprobada por partes** (cada hipótesis con 4 semillas y el color medido en HSV, no a ojo):
1. No es el ControlNet: sin él (escala 0) el pantalón seguía saliendo azul.
2. No son los píxeles de debajo de la máscara: rellenarlos de gris daba la misma imagen.
3. Sí es **la ropa original que queda a la vista**. La máscara no cubre del todo los bordes del jean y el modelo
   continúa esa tela: con solo el prompt «black tailored dress trousers» salía azul. Dilatando la máscara 9 px salió
   negro 4 de 4. Cuando la zona de abajo se genera con la polera original a la vista, copia su azul marino.
4. Además, con las dos prendas en un solo prompt, SD 1.5 mezcla los colores («light blue» tiñe todo el look).

**Solución (approaches.B.dos_pasadas en config/vitrina.yaml).**
- Pasada 1: se tapa todo el look y la zona del pantalón se ensancha 9 px (`dilatar_abajo`), nunca sobre el rostro ni
  el cabello. Se genera con el prompt del pantalón solo y con un negativo «blue, denim, jeans» si el pantalón no es
  azul ni celeste («blue» si es un jean de otro color).
- Pasada 2: se regenera la prenda de arriba con su propio prompt.

Solo se aplica a looks de arriba + abajo separados; ternos y vestidos siguen en una pasada. Cada imagen registra
`pasadas` en sus métricas.

**Resultado** (misma foto, mismas ocasiones):

| | Antes | Después |
|---|---|---|
| Looks con los colores correctos («trabajo» + «casual», 10 looks) | 4 de 10 | 8 de 10 |
| Pantalón negro, en la pasada de abajo, 4 semillas | 0/4 | 4/4 |

El «después» usa 2 candidatos por look elegidos con CLIP (ver abajo) y el «antes» 1, así que la comparación mezcla los
dos cambios. El aislado es el del pantalón negro.

**Candidatos con CLIP en la app local.** `tools/app_local.py` ahora usa 2 candidatos + CLIP si la GPU tiene 6 GB o
más; con 4 GB sigue con 1. CLIP penaliza los estampados que inventa SD (leopardo, cuadros) cuando hay dos pasadas.

**Costo.** Un look de dos prendas tarda el doble (dos pasadas) y con 2 candidatos otra vez el doble: los 5 looks pasan
de unos 50 s a unos 2 min en la RTX 2060.

**Limitaciones que siguen.**
- Las prendas **negras de arriba** (polo negro) salen gris oscuro.
- A veces aparecen texturas en las mangas.
- Una vez la blusa blanca salió gris (copia el color del pantalón que ya está a la vista en la pasada 2).
- El **jean baggy** sigue saliendo recto: la máscara no sale del contorno del pantalón original (ya estaba anotado).

## Otros arreglos de la rama `estilo-usuario-mejoras`
- `estilo._ajustados` (respaldo de la ronda con mi estilo cuando ya no quedan looks nuevos) elegía looks con prendas de
  «No quiero». Por ejemplo, a quien rechazó el gris le daba tres looks con pantalón gris, «porque prefieres corte
  recto». También repetía la misma ropa. Ahora nunca usa algo de «No quiero» si hay alternativa y no repite ropa. Esto
  afecta la métrica «% que respeta No quiero».
- Los notebooks y el ZIP del código salen idénticos en Windows y Linux (antes cambiaban por los saltos de línea y los
  tests decían que estaban desactualizados).
- `tools/app_local.py` guarda las sesiones en `~/PTDIA/sesiones_local` en Linux (en Windows sigue `C:\PTDIA\...`).
