# Guion del video (máximo 3 minutos)

Requisitos del enunciado: video de 3 minutos o menos en el que **una persona externa** (docente o estudiante que no sea de
Computación ni de Ciencia de Datos) usa el sistema. Al inicio, esa persona **da su consentimiento en voz alta, unos 10 segundos**,
para que se grabe su uso y se use su foto.

**Antes de grabar**
- Cierra Chrome y todo lo pesado; abre la app local (`.venv\Scripts\python.exe tools\app_local.py`) y confirma que dice
  «Enfoque API: disponible». Cuida que en pantalla **no aparezca la clave** (`.env`).
- La foto es **de la propia persona participante** (con su consentimiento). Nunca fotos de terceros.
- Graba a 1080p con la ventana del navegador al 110 %. La generación tarda unos 2 minutos: **se corta la espera en la edición**
  (se deja el indicador de la cinta un instante para que se note que está trabajando).

| # | Tiempo | Pantalla | Qué se dice o se hace |
|---|--------|----------|-----------------------|
| 1 | 0:00–0:20 | Cabecera de la app (la cinta métrica) | Problema: comprar ropa en línea sin probársela. Idea: Vitrina IA te muestra cinco looks sobre **tu propia foto**; tú decides y una IA te explica tu elección. |
| 2 | 0:20–0:35 | La participante, a cámara | **Consentimiento (10 s):** «Soy ... y autorizo que se grabe mi uso de este sistema y que se use mi foto en este video.» |
| 3 | 0:35–1:00 | Formulario | La participante sube su foto, marca que es suya y que es mayor de edad, elige la línea y la ocasión. Se muestra «Cómo generar tus looks» con las dos maneras (local y por API). |
| 4 | 1:00–1:20 | Generación | Elige una manera, pulsa «Probarme 5 looks» (si usa la API, marca el consentimiento de enviar la foto). La cinta «mide» mientras espera. Corte. |
| 5 | 1:20–2:00 | Los cinco looks | Aparecen V1–V5. La participante dice qué le gusta y qué no: marca «Me gusta» / «No me gusta»; la cinta de cada look se marca («Aceptado» / «Descartado»). Pide **nueva ronda** solo para los rechazados. |
| 6 | 2:00–2:35 | «Explicar mi elección» | El agente lista qué aceptó y qué rechazó y aparece «verificado contra tu registro». Se menciona que un verificador sin LLM comprueba cada resumen. |
| 7 | 2:35–3:00 | Cierre (voz del equipo sobre la pantalla) | Dos maneras de generar (gratuita y de pago); guardrails (consentimiento, prendas permitidas, sin guardar la foto); limitaciones honestas: prendas descritas, no exactas, y colores que a veces se mezclan en el intermedio. |

**No decir** cifras que no estén en el informe (ventas, devoluciones, satisfacción). Si algo falla en la grabación, repetir la toma:
no editar para ocultar un fallo del sistema.
