# Procedencia de las imágenes de entrada

**Estado:** las imágenes presentes son exactamente las listadas en `MANIFEST.csv` (lo comprueba
`tools/make_zip.py`). Se registran con `tools/register_image.py`.

- Dataset: Synthetic Faces High Quality – Text 2 Image (SFHQ-T2I), versión 1 (2024-10-04), autor David Beniaguev.
- Fuente oficial: https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models
- Repositorio (código): https://github.com/SelfishGene/SFHQ-T2I-dataset
- Licencia declarada en Kaggle: MIT (verificado 2026-10-21). Aviso: ver `LICENSE_MIT.txt`.
- Subconjunto: `a_small_sample_new/`, solo archivos `SDXL_image_*.jpg` cuyo `model_used` en
  `SFHQ_T2I_dataset.csv` sea `SDXL` (generadas vía Stability API, `stable-diffusion-xl-1024-v1-0`).
- Alternativa aprobada: SFHQ parte 3 (CC0). FLUX.1-dev solo como alternativa.
- Cantidad prevista: 10 imágenes (1 para el piloto, 9 de reserva). Selección: `tools/select_images.py`
  (semilla 2026) + revisión visual; registrar en `MANIFEST.csv` el motivo de cada elección.
- Detalle de licencias y condiciones de los generadores: `planificacion/10_verificaciones_y_plan_piloto.md` §1.
