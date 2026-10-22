"""Proyecto 2: sistema generativo interactivo con feedback humano (UTEC).

__version__ identifica la versión del código; run_colab.ipynb exige que coincida con la suya
para no ejecutar nunca un ZIP antiguo (p.ej. con la validación de versión exacta ya retirada).

Los módulos que necesitan torch/diffusers/transformers los importan dentro de las
funciones, para que el resto (esquemas, decisiones, verificador, UI con mock) se
pueda probar en Windows sin GPU ni modelos.
"""

__version__ = "0.5.0"  # 0.5.0: alias, SUS, BD temporal, enlace público opcional y borrado
