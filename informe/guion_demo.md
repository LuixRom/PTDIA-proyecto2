# Guion de la demostración en vivo (máximo 6 minutos)

## Antes de salir al estrado (checklist)
- [ ] **Chrome y programas pesados cerrados**; unos 8 GB de RAM libres (Administrador de tareas, pestaña Rendimiento).
- [ ] `.venv\Scripts\python.exe tools\app_local.py` abierto **desde antes** de empezar (carga modelos); la terminal dice «Enfoque API: disponible» y la pantalla muestra la versión 3.8.0.
- [ ] Saldo en la cuenta de la API (cada imagen cuesta unos 0,023 USD) y la clave **no visible** en pantalla.
- [ ] Foto de demostración **propia o de un integrante con su consentimiento**, en una carpeta a mano. Nunca fotos de terceros.
- [ ] Plan B listo: video del uso (guion_video.md) y una sesión ya generada para mostrar el historial.
- [ ] Un integrante controla el tiempo (aviso a los 4 y a los 5 minutos 30).

## Línea de tiempo
| Tiempo | Qué se muestra | Qué se dice |
|--------|----------------|-------------|
| 0:00–0:30 | Cabecera de la app (la cinta) | El problema (probarse ropa en línea) y la idea: cinco looks sobre tu propia foto; tú decides, una IA explica. |
| 0:30–1:15 | Figura de la arquitectura del informe | Flujo: guardrails de entrada → segmentación y pose → generación (intermedio local o API) → guardrails de salida → revisión → registro → agente + verificador. |
| 1:15–1:45 | App: subir la foto, línea y ocasión | Consentimiento y mayoría de edad; «Cómo generar tus looks»: las dos maneras y por qué existen (gratis y reproducible vs. fiel pero de pago, y la foto sale del equipo). |
| 1:45–3:15 | «Probarme 5 looks» con la API (≈ 2 min) | **Mientras espera**, explicar los guardrails: consentimiento aparte para la API, términos vetados, rostro y cabello originales, etiqueta «Simulación IA», no se guarda la foto. La cinta del encabezado «mide» marca por marca. |
| 3:15–4:00 | Los cinco looks | Aceptar y rechazar en vivo; la cinta de cada look se marca («Aceptado» / «Descartado»). Pedir **nueva ronda** (solo regenera los rechazados). |
| 4:00–4:45 | «Explicar mi elección» | Elegir el LLM (Qwen o Phi en ajustes avanzados); mostrar «verificado contra tu registro» y qué pasa si el verificador no está de acuerdo (plantilla marcada). |
| 4:45–5:15 | Pestañas Historial y Registro | Historial de rondas con semillas (en modo prueba son fijas y restaurables) y registro estructurado de las decisiones; métricas de la sesión (imágenes, costo estimado). |
| 5:15–6:00 | Cierre | Resultados honestos: intermedio (colores mezclados a veces), API (colores correctos, pero de pago y sin semilla), experimento LoRA que **no mejoró**; trabajo pendiente: pruebas con usuarios (SUS) y prenda exacta con IP-Adapter. |

## Si algo falla en vivo
- **La API no responde o no hay saldo:** cambiar a «Intermedio» (gratis) en el mismo selector y seguir; avisar que tarda más.
- **La app se cierra al arrancar (falta de memoria):** pasar al video del uso; no intentar relanzar en vivo.
- **Un look sale mal:** es parte del ejercicio; rechazarlo y pedir nueva ronda, que es justo lo que muestra el sistema.
- **Nunca** mostrar `.env`, la terminal con la clave o la cuenta de OpenAI.
