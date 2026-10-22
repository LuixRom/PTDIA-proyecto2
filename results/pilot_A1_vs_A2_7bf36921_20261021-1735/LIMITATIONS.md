# Limitaciones de esta ejecución (completar tras la evaluación)
- Muestra: 1 imagen de entrada x 2 métodos x 5 variaciones x 1 semilla. Resultados EXPLORATORIOS: no generalizables.
- Las métricas automáticas no miden identidad; CLIP no se usa para reconocer personas.
- Sesgos del generador del dataset (SDXL) y de SD 1.5: anotar cambios no pedidos (tono de piel, edad, rasgos).
- Activaciones del safety checker: ver columna nsfw en metrics.csv.
- Si lpips_available = False en set_metrics.csv, no hay distancia perceptual LPIPS: solo SSIM, CLIP y juicio humano.
- Versiones exactas usadas (p.ej. diffusers 0.41.0 en vez del 0.40.0 probado): ver validation.json y env.json.
- Imagen de entrada sintética (SFHQ-T2I): parecido accidental con personas reales no descartable.
